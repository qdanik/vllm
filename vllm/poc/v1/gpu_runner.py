"""GPU model runner integration for PoC (Proof of Compute)."""

from typing import Any

import numpy as np
import torch

from vllm.v1.core.sched.output import SchedulerOutput


def batch_has_poc(scheduler_output: "SchedulerOutput", requests: dict) -> bool:
    """Check if batch contains any PoC requests."""
    for req_id in scheduler_output.num_scheduled_tokens:
        req_state = requests.get(req_id)
        if req_state is not None and req_state.poc_params is not None:
            return True
    return False


def fill_poc_inputs_embeds(
    scheduler_output: "SchedulerOutput",
    requests: dict,
    input_batch,
    inputs_embeds_gpu: torch.Tensor,
    is_token_ids_gpu: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
    is_first_rank: bool,
    model_config,
) -> None:
    """Fill PoC embeddings into inputs_embeds.gpu for this step.

    This runs on the first PP rank only and relies on InputBatch.is_token_ids
    being False for PoC prompt tokens.
    """
    if not is_first_rank:
        return

    if not batch_has_poc(scheduler_output, requests):
        return

    from vllm.poc.v1.gpu import build_poc_prompt_embeds

    hidden_size = model_config.get_hidden_size()

    # Requests in InputBatch are ordered; scheduled tokens for each request
    # are laid out contiguously in that same order.
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
        embeds = build_poc_prompt_embeds(
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
            # Slice in case any non-zero computed tokens existed (shouldn't for PoC).
            start_pos = int(input_batch.num_computed_tokens_cpu[req_index])
            seg_len = end - start

            # Bounds check
            seq_len = embeds.shape[1]
            if start_pos + seg_len > seq_len:
                end_pos = start_pos + seg_len
                raise ValueError(
                    f"PoC embedding slice out of bounds: {end_pos} > {seq_len}"
                )

            embedding_slice = embeds[row, start_pos : start_pos + seg_len, :]

            # Check for NaN/Inf before reshape
            if embedding_slice.isnan().any():
                msg = (
                    f"PoC embedding contains NaN values! req_index={req_index}, "
                    f"start_pos={start_pos}, seg_len={seg_len}"
                )
                raise RuntimeError(msg)
            if embedding_slice.isinf().any():
                msg = (
                    f"PoC embedding contains Inf values! req_index={req_index}, "
                    f"start_pos={start_pos}, seg_len={seg_len}"
                )
                raise RuntimeError(msg)

            embedding_slice = embedding_slice.reshape(seg_len, -1)

            # Ensure contiguity before copy to avoid CUDA issues
            embedding_slice = embedding_slice.contiguous()

            # Final bounds check before copy
            if start + seg_len > inputs_embeds_gpu.shape[0]:
                buffer_size = inputs_embeds_gpu.shape[0]
                msg = (
                    f"Output buffer out of bounds: [{start}:{end}] "
                    f"exceeds shape {buffer_size}"
                )
                raise ValueError(msg)

            inputs_embeds_gpu[start:end].copy_(embedding_slice)
            # Mark these tokens as embeddings, not token IDs.
            is_token_ids_gpu[start:end] = False


def extract_poc_results(
    scheduler_output: "SchedulerOutput",
    requests: dict,
    input_batch,
    sample_hidden_states: torch.Tensor,
    device: torch.device,
) -> dict[str, dict] | None:
    """Extract PoC computation results from hidden states.

    Must be called immediately after forward pass, before sampling.
    """
    if not batch_has_poc(scheduler_output, requests):
        return None

    from vllm.poc.v1.gpu import compute_poc_result

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
