"""GPU data-plane for PoC: embedding generation and result extraction.

This module merges the responsibilities of the former ``gpu_artifacts``
and ``gpu_model_runner_integration`` modules into a single coherent
data-plane layer.  All functions are called exclusively from
:class:`~vllm.poc.engine.plugin.PoCRunnerPlugin`.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import torch

from vllm.poc.consensus.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.engine.params import PoCSchedulerParams
from vllm.v1.core.sched.output import SchedulerOutput

_EPS_F32: float = 1e-8


def _normalize_rows_f32(x: torch.Tensor, *, eps: float = _EPS_F32) -> torch.Tensor:
    """Row-wise L2 normalisation in fp32."""
    x = x.float()
    denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True).add(eps)
    return x / denom


def _require_non_empty(params: list[PoCSchedulerParams]) -> None:
    if not params:
        raise ValueError("params must be non-empty")


def _assert_same(
    params: list[PoCSchedulerParams],
    *,
    fields: Iterable[str],
) -> None:
    _require_non_empty(params)
    first = params[0]
    for i, p in enumerate(params[1:], start=1):
        for f in fields:
            if getattr(p, f) != getattr(first, f):
                raise ValueError(
                    f"PoCParams group mismatch at index={i}: field={f} "
                    f"({getattr(p, f)!r} != {getattr(first, f)!r})"
                )


@torch.inference_mode()
def build_poc_prompt_embeddings(
    params: list[PoCSchedulerParams],
    *,
    hidden_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate PoC prompt embeddings on GPU for a group of PoC requests.

    Assumes all params share (block_hash, public_key, seq_len).

    Returns: [batch, seq_len, hidden_size]
    """
    _assert_same(params, fields=("block_hash", "public_key", "seq_len"))

    param = params[0]
    nonces = [p.nonce for p in params]

    # generate_inputs must be deterministic for fixed inputs.
    return generate_inputs(
        block_hash=param.block_hash,
        public_key=param.public_key,
        nonces=nonces,
        dim=hidden_size,
        seq_len=param.seq_len,
        device=device,
        dtype=dtype,
    )


@torch.inference_mode()
def compute_poc_result(
    params: list[PoCSchedulerParams],
    *,
    last_hidden: torch.Tensor,
) -> dict[int, dict]:
    """Compute PoC results from hidden states.

    Args:
        params: group of PoC params (same block_hash, public_key, seq_len, k_dim)
        last_hidden: [batch, hidden_size]

    Returns:
        dict[nonce, {"nonces": [...], "vectors_b64": [...]}]
    """
    _assert_same(params, fields=("block_hash", "public_key", "seq_len", "k_dim"))

    param = params[0]
    nonces = [p.nonce for p in params]
    device = last_hidden.device
    k_dim = param.k_dim

    # Normalize hidden states BEFORE subsetting (consensus-critical).
    last_hidden_f32 = _normalize_rows_f32(last_hidden)

    # Subset selection
    pick_idx = random_pick_indices(
        block_hash=param.block_hash,
        public_key=param.public_key,
        nonces=nonces,
        hidden_size=last_hidden_f32.shape[-1],
        k_dim=k_dim,
        device=device,
    )

    # Gather selected indices from normalized hidden states
    x_sub = torch.gather(
        last_hidden_f32,
        dim=-1,
        index=pick_idx.long(),
    )

    # Haar rotation
    x_rot = apply_haar_rotation(
        block_hash=param.block_hash,
        public_key=param.public_key,
        nonces=nonces,
        x=x_sub,
        device=device,
    )

    # Normalise
    x_norm = _normalize_rows_f32(x_rot)
    x_f16 = x_norm.to(torch.float16)

    # Single GPU→CPU transfer for the whole batch, then slice the
    # contiguous numpy buffer.  Avoids per-nonce .tobytes() overhead and
    # keeps the implicit CUDA sync to exactly one call.
    cpu = x_f16.cpu().numpy()
    # Ensure C-contiguous so row slicing is a cheap view.
    if not cpu.flags["C_CONTIGUOUS"]:
        cpu = np.ascontiguousarray(cpu)
    row_bytes = cpu.strides[0]
    raw = cpu.tobytes()
    results: dict[int, dict] = {}
    for i, nonce in enumerate(nonces):
        results[nonce] = {
            "nonces": [nonce],
            "vectors_bin": [raw[i * row_bytes:(i + 1) * row_bytes]],
        }
    return results


def fill_poc_inputs_embeds(
    scheduler_output: SchedulerOutput,
    requests: dict,
    input_batch,
    inputs_embeds_gpu: torch.Tensor,
    is_token_ids_gpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    is_first_rank: bool,
    model_config,
) -> None:
    """Fill PoC embeddings into ``inputs_embeds_gpu`` for this step."""
    if not is_first_rank:
        return

    hidden_size = model_config.get_hidden_size()

    req_ids = input_batch.req_ids
    num_sched = [scheduler_output.num_scheduled_tokens[rid] for rid in req_ids]
    num_sched_np = np.asarray(num_sched, dtype=np.int32)
    cu_num_tokens = np.cumsum(num_sched_np, dtype=np.int32)

    # Group PoC requests by compatible (block_hash, public_key, seq_len, k_dim).
    groups: dict[tuple[str, str, int, int], list[tuple[int, int, int]]] = {}
    params_by_req_index: dict[int, Any] = {}

    for req_index, rid in enumerate(req_ids):
        req_state = requests.get(rid)
        if req_state is None or req_state.poc_params is None:
            continue
        params = req_state.poc_params
        params_by_req_index[req_index] = params
        start = int(cu_num_tokens[req_index] - num_sched_np[req_index])
        end = int(cu_num_tokens[req_index])
        key = (params.block_hash, params.public_key, params.seq_len, params.k_dim)
        groups.setdefault(key, []).append((req_index, start, end))

    for _, items in groups.items():
        params_list = [params_by_req_index[i] for i, _, _ in items]
        embeds = build_poc_prompt_embeddings(
            params_list,
            hidden_size=hidden_size,
            device=device,
            dtype=dtype,
        )
        # Quick validation BEFORE attempting any writes
        max_end = max((end for _, _, end in items), default=0)
        if max_end > inputs_embeds_gpu.shape[0]:
            buffer_size = inputs_embeds_gpu.shape[0]
            raise RuntimeError(
                f"PoC write position {max_end} exceeds buffer size {buffer_size}"
            )

        for row, (req_index, start, end) in enumerate(items):
            start_pos = int(input_batch.num_computed_tokens_cpu[req_index])
            seg_len = end - start
            inputs_embeds_gpu[start:end].copy_(
                embeds[row, start_pos : start_pos + seg_len]
            )
            is_token_ids_gpu[start:end] = False


def extract_poc_results(
    scheduler_output: SchedulerOutput,
    requests: dict,
    input_batch,
    sample_hidden_states: torch.Tensor,
    device: torch.device,
) -> dict[str, dict] | None:
    """Extract PoC computation results from hidden states."""
    req_ids = input_batch.req_ids
    poc_indices: list[int] = []
    poc_params_list: list[Any] = []

    for i, rid in enumerate(req_ids):
        st = requests.get(rid)
        if st is not None and st.poc_params is not None:
            poc_indices.append(i)
            poc_params_list.append(st.poc_params)

    if not poc_indices:
        return None

    # Group by compatible params.
    groups: dict[tuple[str, str, int, int], list[int]] = {}
    for idx, params in zip(poc_indices, poc_params_list):
        key = (params.block_hash, params.public_key, params.seq_len, params.k_dim)
        groups.setdefault(key, []).append(idx)

    poc_results = {}
    for _, indices in groups.items():
        params_group = [requests[req_ids[i]].poc_params for i in indices]
        last_hidden = sample_hidden_states[torch.tensor(indices, device=device)]
        res_by_nonce = compute_poc_result(params_group, last_hidden=last_hidden)
        for i in indices:
            nonce = requests[req_ids[i]].poc_params.nonce
            poc_results[req_ids[i]] = res_by_nonce[nonce]

    return poc_results
