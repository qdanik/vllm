"""Worker-side legacy_poc forward pass for vLLM 0.17.0.

Called via ``collective_rpc_async`` from the async server, bypassing the
scheduler.  The full transformer model (all layers + Householder hooks)
is executed identically to the scheduler path; only the KV-cache write is
skipped via ``direct_qkv=True`` in the attention metadata.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm.distributed import get_pp_group, get_tp_group
from vllm.distributed.communication_op import broadcast_tensor_dict
from vllm.forward_context import set_forward_context
import vllm.poc.env as env
from vllm.poc.consensus.hooks import poc_forward_context
from vllm.poc.consensus.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.engine.gpu import _normalize_rows_f32
from vllm.sequence import IntermediateTensors

# PAD_SLOT_ID = -1 in v1; all-negative slot_mapping triggers direct_qkv=True
_PAD_SLOT_ID: int = -1

# ---------------------------------------------------------------------------
# Module-level caches (survive across collective_rpc calls for the process)
# ---------------------------------------------------------------------------

_attn_meta_cache: dict[tuple, Any] = {}
_positions_cache: dict[tuple, torch.Tensor] = {}


def _get_positions_flat(
    batch_size: int, seq_len: int, device: torch.device
) -> torch.Tensor:
    key = (batch_size, seq_len, str(device))
    if key not in _positions_cache:
        p = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        _positions_cache[key] = p.reshape(-1)
    return _positions_cache[key]


def _build_legacy_poc_attn_metadata(
    batch_size: int, seq_len: int, device: torch.device
) -> Any:
    """Build v1 FlashAttentionMetadata for full-prefill legacy_poc.

    The critical field is ``direct_qkv=True``, which tells
    ``FlashAttentionImpl.forward`` to run causal self-attention over the
    provided Q/K/V directly, without touching or writing to KV-cache blocks.
    """
    from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata

    num_tokens = batch_size * seq_len
    query_start_loc = (
        torch.arange(batch_size + 1, dtype=torch.int32, device=device) * seq_len
    )
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)
    # All -1 → slot_mapping_actual < 0 → direct_qkv=True (skip KV writes)
    slot_mapping = torch.full(
        (num_tokens,), _PAD_SLOT_ID, dtype=torch.long, device=device
    )
    # Empty block_table: no KV blocks allocated
    block_table = torch.zeros((batch_size, 0), dtype=torch.int32, device=device)

    return FlashAttentionMetadata(
        num_actual_tokens=num_tokens,
        max_query_len=seq_len,
        query_start_loc=query_start_loc,
        max_seq_len=seq_len,
        seq_lens=seq_lens,
        block_table=block_table,
        slot_mapping=slot_mapping,
        use_cascade=False,
        common_prefix_len=0,
        cu_prefix_query_lens=None,
        prefix_kv_lens=None,
        suffix_kv_lens=None,
        causal=True,
        direct_qkv=True,  # bypass KV cache — the key legacy_poc optimisation
    )


def _get_legacy_poc_attn_metadata(
    batch_size: int, seq_len: int, device: torch.device
) -> Any:
    key = (batch_size, seq_len, str(device))
    if key not in _attn_meta_cache:
        _attn_meta_cache[key] = _build_legacy_poc_attn_metadata(
            batch_size, seq_len, device
        )
    return _attn_meta_cache[key]


def _ensure_legacy_poc_hooks(worker: Any, block_hash: str) -> None:
    """Delegate Householder hook management to the scheduler plugin."""
    worker.model_runner._poc._ensure_hooks(block_hash)


# ---------------------------------------------------------------------------
# Main worker function — called via collective_rpc
# ---------------------------------------------------------------------------


@torch.inference_mode()
def execute_legacy_poc_forward_multi_batch(
    worker: Any,
    block_hash: str,
    public_key: str,
    all_nonces: list[int],
    batch_size: int,
    seq_len: int,
    hidden_size: int,
    k_dim: int,
) -> dict[str, Any] | None:
    """Execute multiple PoC forward passes in one ``collective_rpc`` call."""
    device: torch.device = worker.model_runner.device
    dtype: torch.dtype = worker.model_runner.dtype
    vllm_config = worker.model_runner.vllm_config
    model = worker.model_runner.get_model()

    tp_group = get_tp_group()
    pp_group = get_pp_group()

    # =========================================================================
    # TP SYNC: rank-0 driver broadcasts parameters to all TP workers.
    # Non-driver ranks cannot run generate_inputs without these values.
    # =========================================================================
    if tp_group.world_size > 1:
        if tp_group.rank_in_group == 0:
            broadcast_tensor_dict(
                {
                    "seq_len": seq_len,
                    "hidden_size": hidden_size,
                    "all_nonces": all_nonces,
                    "batch_size": batch_size,
                    "k_dim": k_dim,
                },
                src=0,
            )
        else:
            data = broadcast_tensor_dict(src=0)
            seq_len = int(data["seq_len"])
            hidden_size = int(data["hidden_size"])
            all_nonces = list(data["all_nonces"])
            batch_size = int(data["batch_size"])
            k_dim = int(data["k_dim"])

    total = len(all_nonces)

    # Install (or reuse) Householder layer hooks for this block_hash.
    _ensure_legacy_poc_hooks(worker, block_hash)

    # Pre-fetch cached attention metadata and flat position tensors.
    attn_meta = _get_legacy_poc_attn_metadata(batch_size, seq_len, device)
    positions = _get_positions_flat(batch_size, seq_len, device)

    # Handle a potential last mini-batch smaller than batch_size.
    last_bs = total % batch_size
    if last_bs > 0 and last_bs != batch_size:
        attn_meta_last = _get_legacy_poc_attn_metadata(last_bs, seq_len, device)
        positions_last = _get_positions_flat(last_bs, seq_len, device)
    else:
        attn_meta_last = attn_meta
        positions_last = positions

    all_vectors: np.ndarray | None = None
    all_result_nonces: list[int] = []

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        nonces = all_nonces[start:end]
        cur_bs = len(nonces)

        batch_attn = attn_meta if cur_bs == batch_size else attn_meta_last
        batch_pos = positions if cur_bs == batch_size else positions_last

        # ── First PP rank: generate crypto embeddings ──────────────────────
        intermediate_tensors: IntermediateTensors | None = None
        inputs_embeds: torch.Tensor | None = None

        if pp_group.is_first_rank:
            inputs_embeds = generate_inputs(
                block_hash=block_hash,
                public_key=public_key,
                nonces=nonces,
                dim=hidden_size,
                seq_len=seq_len,
                device=device,
                dtype=dtype,
            )
        else:
            intermediate_tensors = IntermediateTensors(
                pp_group.recv_tensor_dict(all_gather_group=tp_group)
            )

        # ── Model forward (all layers + Householder hooks) ──────────────────
        with (
            set_forward_context(batch_attn, vllm_config, skip_compiled=True),
            poc_forward_context(apply_all=True),
        ):
            num_tokens = cur_bs * seq_len
            # For AOT-compiled models, input_ids cannot be None even with
            # inputs_embeds provided. Use zeros placeholder if workaround enabled.
            if env.POC_USE_AOT_COMPILED_WORKAROUND:
                input_ids = torch.zeros(
                    num_tokens, dtype=torch.long, device=device
                )
            else:
                input_ids = None
            hidden = model(
                input_ids=input_ids,
                positions=batch_pos[:num_tokens],
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=(
                    inputs_embeds.view(-1, hidden_size)
                    if inputs_embeds is not None
                    else None
                ),
            )

        # ── Pipeline pass-through for non-last PP ranks ────────────────────
        if not pp_group.is_last_rank:
            if isinstance(hidden, IntermediateTensors):
                pp_group.send_tensor_dict(
                    hidden.tensors, all_gather_group=tp_group
                )
            continue

        # ── Post-processing (last PP rank only) ────────────────────────────
        # Extract last-token hidden state: [cur_bs, hidden_size].
        last_hidden = hidden.view(cur_bs, seq_len, -1)[:, -1, :]
        last_hidden_f32 = _normalize_rows_f32(last_hidden)

        # Per-nonce k-dim selection + Haar rotation.
        indices = random_pick_indices(
            block_hash, public_key, nonces, hidden_size, k_dim, device
        )
        xk = torch.gather(last_hidden_f32, 1, indices.long())
        yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
        yk = _normalize_rows_f32(yk)

        # Single GPU→CPU transfer.
        yk_cpu = yk.half().cpu().numpy()
        if all_vectors is None:
            all_vectors = np.empty((total, k_dim), dtype=np.float16)
        all_vectors[start:end] = yk_cpu
        all_result_nonces.extend(nonces)

    # Non-last PP ranks return None.
    if not pp_group.is_last_rank:
        return None

    # Serialize vectors as raw bytes so they survive ZMQ/msgpack transport
    # (msgpack silently converts numpy arrays to Python lists on the other end).
    vec_arr = all_vectors if all_vectors is not None else np.empty((0, k_dim), dtype=np.float16)
    return {
        "nonces": all_result_nonces,
        "vectors_bytes": vec_arr.tobytes(),
        "n": len(all_result_nonces),
        "k_dim": k_dim,
    }
