"""PoC model runner for vLLM 0.15.x V1 architecture.

Full model forward pass with proper V1 attention metadata.
Uses actual KV cache blocks for attention to work correctly.
Batched forward pass — by default processes all nonces in a single forward
call. Pass ``batch_size > 0`` (env ``POC_BATCH_SIZE``) to cap the per-forward
chunk size when KV cache can't hold the full nonce list at once; the runner
then loops over chunks and concatenates results.

``skip_compiled=True`` keeps PoC on the eager path even when the rest of the
server uses torch.compile / CUDA graphs (regular ``execute_model`` doesn't
pass this flag), so PoC artifacts stay GPU-portable across Hopper/Blackwell
while normal inference keeps its AOT throughput.
"""
import math
import os
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

logger = init_logger(__name__)

DEFAULT_K_DIM = 12
# 0 = process all nonces in one forward (HEAD default behaviour).
# >0 = cap per-forward chunk size; runner loops if total nonces > cap.
DEFAULT_BATCH_SIZE = int(os.getenv("POC_BATCH_SIZE", "0"))

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


def _forward_chunk(
    worker,
    block_hash: str,
    public_key: str,
    chunk_nonces: List[int],
    seq_len: int,
    hidden_size: int,
    k_dim: int,
    poc_stronger_rng: bool,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
    model,
    vllm_config,
    pp_group,
) -> Optional[Dict[str, Any]]:
    """Run one forward pass over ``len(chunk_nonces)`` sequences.

    Returns dict with ``nonces`` (post-NaN-filter) and ``vectors`` (FP16 numpy)
    on the last PP rank, or ``None`` on intermediate PP ranks.
    """
    cur_bs = len(chunk_nonces)

    # Profile mode — gated by POC_PROFILE=1 env. Adds torch.cuda.synchronize
    # between segments for accurate wallclock timing of GPU work.
    _profile = os.getenv("POC_PROFILE", "0") == "1"
    _times: List[tuple] = []
    def _ck(label: str):
        if _profile:
            torch.cuda.synchronize()
            import time
            _times.append((label, time.perf_counter()))
    _ck("entry")

    attn_metadata, slot_mapping_dict = _get_or_create_attn_metadata(
        cur_bs, seq_len, block_size, device, worker
    )
    _ck("attn_meta")

    # Positions for the batch
    positions = torch.arange(seq_len, device=device).repeat(cur_bs)
    _ck("positions")

    intermediate_tensors = None
    inputs_embeds = None

    if pp_group.is_first_rank:
        kv_caches = getattr(worker.model_runner, "kv_caches", [])
        kv_scratch = None
        needed_elems = cur_bs * seq_len * hidden_size
        for kv in kv_caches:
            # Skip FP8 / uint8-storage KV caches: their byte storage can't be
            # reinterpreted as inputs_embeds without corrupting the floating
            # value (per_token_group_quant would then crash on a Byte tensor).
            if kv.dtype != dtype:
                continue
            if kv.numel() >= needed_elems:
                kv_scratch = kv.flatten()[:needed_elems].view(
                    cur_bs, seq_len, hidden_size)
                break
        if kv_scratch is not None:
            # Batched fill: collapse the per-nonce _normal loop into a single
            # _batched_normal call. For bs=64, seq_len=1024, hidden=4096 this
            # turns 64 sequential kernel launches (with alloc/free thrashing)
            # into one big launch. Result is bit-identical because the
            # batched murmur3 produces the same hash sequence per row.
            from .gpu_random import _seed_from_string, _batched_normal
            seeds = [
                _seed_from_string(f"{block_hash}_{public_key}_nonce{nonce}")
                for nonce in chunk_nonces
            ]
            all_vals = _batched_normal(seeds, seq_len * hidden_size, device)
            kv_scratch.copy_(
                all_vals.view(cur_bs, seq_len, hidden_size).to(dtype)
            )
            del all_vals
            inputs_embeds = kv_scratch
        else:
            _gen_fn = generate_inputs_concat_murmur if poc_stronger_rng else generate_inputs
            inputs_embeds = _gen_fn(
                block_hash, public_key, chunk_nonces,
                dim=hidden_size, seq_len=seq_len,
                device=device, dtype=dtype,
            )
    else:
        intermediate_tensors = IntermediateTensors(
            pp_group.recv_tensor_dict(all_gather_group=get_tp_group())
        )
    _ck("inputs_ready")

    with set_forward_context(
        attn_metadata, vllm_config,
        num_tokens=cur_bs * seq_len,
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
    _ck("model_forward")

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
    hidden_states = hidden_states.view(cur_bs, seq_len, -1)
    last_hidden = hidden_states[:, -1, :].float()  # [cur_bs, hidden_size]

    # NaN detection
    nan_mask = torch.isnan(last_hidden).any(dim=-1)  # [cur_bs]
    chunk_nonces_filtered = chunk_nonces
    if nan_mask.any():
        clean_idx = (~nan_mask).nonzero(as_tuple=True)[0]
        nan_count = nan_mask.sum().item()
        logger.warning("NaN in %d/%d hidden states (GPU fault?)", nan_count, cur_bs)

        if clean_idx.numel() == 0:
            logger.error("All %d nonces produced NaN — chunk rejected", cur_bs)
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

        last_hidden = last_hidden[clean_idx]
        chunk_nonces_filtered = [chunk_nonces[i] for i in clean_idx.tolist()]

    # Normalize to unit sphere
    last_hidden = last_hidden / (last_hidden.norm(dim=-1, keepdim=True) + 1e-8)

    # Batched k-dim pick + Haar rotation
    indices = random_pick_indices(block_hash, public_key, chunk_nonces_filtered, hidden_size, k_dim, device)
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, chunk_nonces_filtered, xk, device)

    # Normalize output vectors
    yk = yk / (yk.norm(dim=-1, keepdim=True) + 1e-8)

    # Convert to FP16
    vectors_f16 = yk.half().cpu().numpy()  # [cur_bs, k_dim]

    # Late NaN check after FP16 conversion
    nan_out = np.isnan(vectors_f16).any(axis=1)
    if nan_out.any():
        clean = ~nan_out
        vectors_f16 = vectors_f16[clean]
        chunk_nonces_filtered = [n for n, c in zip(chunk_nonces_filtered, clean) if c]
        logger.warning("NaN in FP16 output — %d nonces filtered", nan_out.sum())

    _ck("post_processing")
    if _profile and len(_times) >= 2:
        total_ms = (_times[-1][1] - _times[0][1]) * 1000.0
        parts = []
        for i in range(1, len(_times)):
            seg_ms = (_times[i][1] - _times[i-1][1]) * 1000.0
            pct = (seg_ms / total_ms * 100.0) if total_ms > 0 else 0.0
            parts.append(f"{_times[i][0]}={seg_ms:.1f}ms({pct:.1f}%)")
        logger.info("[POC_PROFILE] bs=%d total=%.1fms | %s", cur_bs, total_ms, " ".join(parts))

    return {
        "nonces": chunk_nonces_filtered,
        "vectors": vectors_f16,
    }


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
    batch_size: int = 0,
) -> Optional[Dict[str, Any]]:
    """Execute batched PoC forward pass on a V1 worker.

    By default (``batch_size=0``) processes all nonces in a single forward
    call — maximum throughput, minimum overhead. Pass ``batch_size > 0`` to
    cap the per-forward chunk size; the runner then loops over chunks and
    concatenates results. Useful when KV cache can't hold the full nonce
    list at once.
    """
    device = worker.device
    dtype = worker.model_config.dtype
    model = worker.model_runner.model
    vllm_config = worker.vllm_config

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
                "batch_size": batch_size,
            }, src=0)
        else:
            broadcast_data = broadcast_tensor_dict(src=0)
            seq_len = int(broadcast_data["seq_len"])
            hidden_size = int(broadcast_data["hidden_size"])
            nonces = list(broadcast_data["nonces"])
            k_dim = int(broadcast_data["k_dim"])
            poc_stronger_rng = bool(broadcast_data["poc_stronger_rng"])
            batch_size = int(broadcast_data["batch_size"])

    pp_group = get_pp_group()

    # Pre-forward sync
    if tp_group.world_size > 1:
        dist.barrier(group=tp_group.cpu_group)
    torch.cuda.synchronize()

    _ensure_layer_hooks(worker, block_hash, hidden_size)
    block_size = _get_block_size(worker)

    total = len(nonces)
    if total == 0:
        if pp_group.is_last_rank:
            return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}
        return None

    # batch_size=0 → all in one chunk (HEAD's original behaviour).
    # batch_size>0 → cap and loop.
    chunk_size = batch_size if batch_size > 0 else total
    chunk_size = max(1, min(chunk_size, total))

    all_vectors_chunks: List[np.ndarray] = []
    all_nonces_out: List[int] = []

    for start in range(0, total, chunk_size):
        chunk = nonces[start:start + chunk_size]
        chunk_result = _forward_chunk(
            worker=worker,
            block_hash=block_hash,
            public_key=public_key,
            chunk_nonces=chunk,
            seq_len=seq_len,
            hidden_size=hidden_size,
            k_dim=k_dim,
            poc_stronger_rng=poc_stronger_rng,
            block_size=block_size,
            device=device,
            dtype=dtype,
            model=model,
            vllm_config=vllm_config,
            pp_group=pp_group,
        )
        # Non-last PP rank already sent intermediate; nothing to collect.
        if chunk_result is None:
            continue
        all_vectors_chunks.append(chunk_result["vectors"])
        all_nonces_out.extend(chunk_result["nonces"])

    if not pp_group.is_last_rank:
        return None

    if not all_vectors_chunks:
        return {"nonces": [], "vectors": np.empty((0, k_dim), dtype=np.float16)}

    vectors_f16 = np.concatenate(all_vectors_chunks, axis=0)
    return {
        "nonces": all_nonces_out,
        "vectors": vectors_f16,
    }
