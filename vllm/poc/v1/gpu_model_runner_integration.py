"""GPU model runner integration helpers for PoC (Proof of Compute).

These functions contain the *data-plane* logic for PoC embedding
generation and result extraction.  They are called exclusively by
:class:`~vllm.poc.v1.runner_plugin.PoCRunnerPlugin`, which guarantees
that the current batch actually contains PoC requests before invoking
them — no redundant guards here.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from vllm.v1.core.sched.output import SchedulerOutput


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
    """Fill PoC embeddings into `inputs_embeds_gpu` for this step."""
    if not is_first_rank:
        return

    from vllm.poc.v1.gpu_artifacts import build_poc_prompt_embeddings

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
    from vllm.poc.v1.gpu_artifacts import compute_poc_result

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
