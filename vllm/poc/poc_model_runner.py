"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — processes all nonces in a single forward call.
"""
import math
import torch
import torch.distributed as dist
import numpy as np
from typing import List, Optional, Dict, Any

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
from vllm.sequence import IntermediateTensors
from vllm.logger import init_logger

from .gpu_random import (
    generate_inputs,
    generate_inputs_concat_murmur,
    random_pick_indices,
    apply_haar_rotation,
)
from .layer_hooks import LayerHouseholderHook, poc_forward_context
from .fingerprint_hooks import RoutingFingerprintHook
from .fingerprint.schema import NonceFingerprint
from .fingerprint.site_selection import (
    pick_logit_positions,
    pick_routing_sites,
)
from .fingerprint.logit_capture import capture_seeded_logits

logger = init_logger(__name__)

# Default seeded-site sampling for the Phase-0 fingerprint capture. These are
# capture knobs only (architecture.md Section 5.6 sweeps thresholds offline);
# they are not consensus parameters.
DEFAULT_N_ROUTING_LAYERS_SAMPLE = 8
DEFAULT_N_ROUTING_POSITIONS_SAMPLE = 4
DEFAULT_N_LOGIT_POSITIONS = 4
DEFAULT_FINGERPRINT_TOP_K = 8

DEFAULT_K_DIM = 12

# NOTE: attention metadata must NOT be cached across PoC calls.
# The metadata builder's internal state (workspace buffers, page-table
# references) is mutated by every inference engine step.  Reusing a
# stale metadata object causes the attention backend to write only a
# fraction of the expected KV entries, producing all-NaN hidden states.
# The cost of rebuilding is <1 ms per call (vs ~15 ms for the model
# forward), so the overhead is negligible.


def _ensure_layer_hooks(worker, block_hash, hidden_size):
    """Ensure layer hooks are installed for the given block_hash."""
    model = worker.model_runner.model
    device = worker.device
    existing_hook = getattr(worker, "_poc_layer_hooks", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return
        existing_hook.detach()
    hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
    hook._setup(model, block_hash, device, hidden_size)
    worker._poc_layer_hooks = hook


def _ensure_routing_fingerprint_hook(worker, block_hash):
    """Ensure the routing fingerprint hook is installed for ``block_hash``.

    Mirrors :func:`_ensure_layer_hooks`: reinstalls only when the block_hash
    changes (detaching the prior hooks first). Returns the active hook so the
    caller can drain it after the forward.
    """
    model = worker.model_runner.model
    existing_hook = getattr(worker, "_poc_routing_fingerprint_hook", None)
    if existing_hook is not None:
        if existing_hook.block_hash == block_hash:
            return existing_hook
        existing_hook.detach()
    hook = RoutingFingerprintHook(block_hash)
    hook._setup(model)
    worker._poc_routing_fingerprint_hook = hook
    return hook


def _detach_routing_fingerprint_hook(worker):
    """Detach any installed routing fingerprint hook (used when flag is off)."""
    existing_hook = getattr(worker, "_poc_routing_fingerprint_hook", None)
    if existing_hook is not None:
        existing_hook.detach()
        worker._poc_routing_fingerprint_hook = None


def _build_fingerprints(
    block_hash,
    public_key,
    nonces,
    seq_len,
    num_moe_layers,
    routing_decisions,
    logit_decisions_per_nonce,
):
    """Assemble per-nonce fingerprints from drained routing + logit decisions.

    Applies the seeded site selection (anti-cherry-pick): only routing decisions
    whose ``(layer_idx, position_idx)`` is in the seeded routing-site set for
    that nonce are kept; logit decisions are already restricted to seeded
    positions by the caller. ALL surviving decisions are emitted (no margin
    pre-filter) so the offline analytics can sweep the threshold from 0.

    Args:
        routing_decisions: list of ``(nonce_idx, position_idx, CapturedDecision)``
            from :meth:`RoutingFingerprintHook.drain`. ``layer_idx`` lives in
            ``decision.site_id[0]``.
        logit_decisions_per_nonce: map ``nonce_idx -> [CapturedDecision]``.

    Returns:
        dict ``nonce_id -> NonceFingerprint``.
    """
    # Precompute the allowed routing site set per nonce index.
    allowed_routing_sites = {}
    for nonce_idx, nonce in enumerate(nonces):
        sites = pick_routing_sites(
            block_hash,
            public_key,
            nonce,
            num_moe_layers=num_moe_layers,
            seq_len=seq_len,
            n_layers_sample=DEFAULT_N_ROUTING_LAYERS_SAMPLE,
            n_positions_sample=DEFAULT_N_ROUTING_POSITIONS_SAMPLE,
        )
        allowed_routing_sites[nonce_idx] = set(sites)

    fingerprints = {
        nonce: NonceFingerprint(nonce_id=nonce, decisions=[])
        for nonce in nonces
    }

    for nonce_idx, position_idx, decision in routing_decisions:
        if nonce_idx >= len(nonces):
            continue
        layer_idx = decision.site_id[0]
        if (layer_idx, position_idx) not in allowed_routing_sites[nonce_idx]:
            continue
        fingerprints[nonces[nonce_idx]].decisions.append(decision)

    for nonce_idx, decisions in logit_decisions_per_nonce.items():
        if nonce_idx >= len(nonces):
            continue
        fingerprints[nonces[nonce_idx]].decisions.extend(decisions)

    return fingerprints


def _serialize_fingerprints(fingerprints_dict):
    """Convert ``{nonce_id: NonceFingerprint}`` to plain picklable data.

    Returns ``{nonce_id: [decision_dict, ...]}`` using the exact decision-dict
    shape the analytics ``load_dump`` reads, so the harness can pass it straight
    to :func:`vllm.poc.fingerprint.schema.to_dump_jsonl` (after rebuilding the
    dataclasses) or dump it directly. ``None`` maps to an empty dict.
    """
    if not fingerprints_dict:
        return {}
    return {
        nonce_id: fingerprint.to_decision_dicts()
        for nonce_id, fingerprint in fingerprints_dict.items()
    }


def _capture_fingerprints(
    model,
    block_hash,
    public_key,
    nonces,
    seq_len,
    hidden_states_3d,
    routing_hook,
):
    """Drain routing decisions, capture seeded logits, and build fingerprints.

    Returns a dict ``nonce_id -> NonceFingerprint``.
    """
    routing_decisions = routing_hook.drain()
    # The routing site-selection layer-index space is the set of hooked FusedMoE
    # modules, enumerated in named_modules order.
    num_moe_layers = routing_hook.num_layers

    # Seeded logit positions per nonce (anti-cherry-pick), then one cheap LM-head
    # matmul over only those rows.
    seeded_positions_per_nonce = {
        nonce_idx: pick_logit_positions(
            block_hash,
            public_key,
            nonce,
            seq_len=seq_len,
            n_positions=DEFAULT_N_LOGIT_POSITIONS,
        )
        for nonce_idx, nonce in enumerate(nonces)
    }
    logit_decisions_per_nonce = capture_seeded_logits(
        model,
        hidden_states_3d,
        seeded_positions_per_nonce,
        top_k=DEFAULT_FINGERPRINT_TOP_K,
    )

    return _build_fingerprints(
        block_hash,
        public_key,
        nonces,
        seq_len,
        num_moe_layers,
        routing_decisions,
        logit_decisions_per_nonce,
    )


def _get_block_size(worker):
    """Get the KV cache block size from the worker config."""
    return worker.cache_config.block_size


def _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create attention metadata for batch_size sequences.

    Uses the worker's metadata builders to create the correct metadata
    for whatever attention backend is configured (FlashAttention,
    FlashInfer, etc.).
    """
    from vllm.v1.attention.backend import CommonAttentionMetadata

    blocks_per_seq = math.ceil(seq_len / block_size)
    total_tokens = batch_size * seq_len

    # slot_mapping: each sequence gets its own block range
    all_slots = []
    for seq_idx in range(batch_size):
        base_block = seq_idx * blocks_per_seq
        for t in range(seq_len):
            block_idx = base_block + t // block_size
            all_slots.append(block_idx * block_size + t % block_size)
    slot_mapping = torch.tensor(all_slots, dtype=torch.long, device=device)

    # block_table: [batch_size, blocks_per_seq]
    block_table = torch.arange(
        batch_size * blocks_per_seq, dtype=torch.int32, device=device
    ).view(batch_size, blocks_per_seq)

    # query_start_loc: [0, seq_len, 2*seq_len, ..., batch_size*seq_len]
    query_start_loc_gpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    query_start_loc_cpu = (
        torch.arange(batch_size + 1, dtype=torch.int32, device="cpu") * seq_len
    )

    seq_lens_gpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device=device
    )
    seq_lens_cpu = torch.full(
        (batch_size,), seq_len, dtype=torch.int32, device="cpu"
    )

    common_attn_metadata = CommonAttentionMetadata(
        query_start_loc=query_start_loc_gpu,
        query_start_loc_cpu=query_start_loc_cpu,
        seq_lens=seq_lens_gpu,
        num_reqs=batch_size,
        num_actual_tokens=total_tokens,
        max_query_len=seq_len,
        max_seq_len=seq_len,
        block_table_tensor=block_table,
        slot_mapping=slot_mapping,
        causal=True,
        _seq_lens_cpu=seq_lens_cpu,
        seq_lens_cpu_upper_bound=seq_lens_cpu,
        _num_computed_tokens_cpu=torch.zeros(
            batch_size, dtype=torch.int32, device="cpu"
        ),
    )

    model_runner = worker.model_runner
    attn_metadata_dict = {}
    slot_mapping_dict = {}

    for kv_cache_group_attn_groups in model_runner.attn_groups:
        for attn_group in kv_cache_group_attn_groups:
            builder = attn_group.get_metadata_builder(0)
            metadata = builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
            )
            for layer_name in attn_group.layer_names:
                attn_metadata_dict[layer_name] = metadata
                slot_mapping_dict[layer_name] = slot_mapping

    return attn_metadata_dict, slot_mapping_dict


def _get_or_create_attn_metadata(batch_size, seq_len, block_size, device, worker):
    """Create fresh attention metadata for the given parameters."""
    return _create_v1_attn_metadata(batch_size, seq_len, block_size, device, worker)


@torch.inference_mode()
def execute_poc_forward(
    worker,
    block_hash: str,
    public_key: str,
    nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int = DEFAULT_K_DIM,
    poc_stronger_rng: bool = False,
    capture_fingerprint: bool = False,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    Processes all nonces in a single forward call for maximum throughput.

    When ``capture_fingerprint`` is True (Phase-0 calibration flag,
    architecture.md Section 5.7), additionally capture the discrete
    decision-boundary fingerprint (seeded per-(layer, position) expert-routing
    ids + margins, and seeded-position top-k logit ids + margins) and return it
    under the ``"fingerprints"`` key. When False, behaviour is unchanged: no
    hook is installed and no extra compute runs.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config
    batch_size = len(nonces)

    tp_group = get_tp_group()
    is_tp_driver = tp_group.rank_in_group == 0

    # TP SYNC
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
        if is_tp_driver:
            broadcast_tensor_dict({
                "poc_go": True,
                "seq_len": seq_len,
                "hidden_size": hidden_size,
                "nonces": nonces,
                "k_dim": k_dim,
                "poc_stronger_rng": poc_stronger_rng,
                "capture_fingerprint": capture_fingerprint,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            batch_size = len(nonces)
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            capture_fingerprint = bool(broadcast_data["capture_fingerprint"])

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)

    # Routing fingerprint capture (Phase-0 flag). Install per block_hash; when
    # the flag is off, detach any prior hook so there is zero capture overhead.
    routing_fingerprint_hook = None
    if capture_fingerprint:
        routing_fingerprint_hook = _ensure_routing_fingerprint_hook(
            worker, block_hash
        )
        routing_fingerprint_hook.set_seq_len(seq_len)
    else:
        _detach_routing_fingerprint_hook(worker)

    # Get block_size and prepare attention metadata (cached, reused)
    block_size = _get_block_size(worker)
    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        batch_size, seq_len, block_size, device, worker
    )

    # Positions for the batch
    positions = torch.arange(seq_len, device=device).repeat(batch_size)

    # Generate inputs for all nonces at once
    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        kv_caches = getattr(worker.model_runner, "kv_caches", [])
        kv_scratch = None
        needed_elems = batch_size * seq_len * hidden_size
        for kv in kv_caches:
            if kv.numel() >= needed_elems:
                kv_scratch = kv.flatten()[:needed_elems].view(
                    batch_size, seq_len, hidden_size)
                break
        if kv_scratch is not None:
            from .gpu_random import _seed_from_string, _normal
            for i, nonce in enumerate(nonces):
                seed = _seed_from_string(
                    f"{block_hash}_{public_key}_nonce{nonce}")
                vals = _normal(seed, seq_len * hidden_size, device)
                kv_scratch[i].copy_(vals.view(seq_len, hidden_size).to(dtype))
                del vals
            inputs_embeds = kv_scratch
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            inputs_embeds = _gen_fn(
                block_hash, public_key, nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )

    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=batch_size * seq_len,
        slot_mapping=slot_mapping_dict,
        skip_compiled=True,
    ):
        with poc_forward_context():
            hidden_states = model(
                input_ids=None,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds.view(-1, hidden_size) if inputs_embeds is not None else None,
            )

    # PP: send to next rank if not last
    if not pp_group.is_last_rank:
        if isinstance(hidden_states, IntermediateTensors):
            pp_group.send_tensor_dict(
                hidden_states.tensors, all_gather_group=get_tp_group()
            )
        return None

    # Handle tuple return
    if isinstance(hidden_states, tuple):
        hidden_states = hidden_states[0]

    # Extract last hidden per sequence
    hidden_states = hidden_states.view(batch_size, seq_len, -1)

    # Phase-0 fingerprint capture (flagged). Uses the full pre-NaN-filter nonce
    # list and the 3-D hidden states; routing decisions were buffered by the
    # forward_pre_hook during the forward above.
    fingerprints_dict = None
    if capture_fingerprint and routing_fingerprint_hook is not None:
        fingerprints_dict = _capture_fingerprints(
            model=model,
            block_hash=block_hash,
            public_key=public_key,
            nonces=list(nonces),
            seq_len=seq_len,
            hidden_states_3d=hidden_states,
            routing_hook=routing_fingerprint_hook,
        )

    last_hidden = hidden_states[:, -1, :].float()  # [batch_size, hidden_size]

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [batch_size]
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, batch_size)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — batch rejected", batch_size)
            result = {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}
            if capture_fingerprint:
                result["fingerprints"] = _serialize_fingerprints(fingerprints_dict)
            return result

        last_hidden = last_hidden[clean_idx]
        nonces = [nonces[i] for i in clean_idx.tolist()]
        batch_size = len(nonces)

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, nonces, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [batch_size, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        nonces = [n for n, c in zip(nonces, clean) if c]
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    result = {
        "nonces": nonces,
        "vectors": vectors_f16,
    }
    if capture_fingerprint:
        result["fingerprints"] = _serialize_fingerprints(fingerprints_dict)
    return result
