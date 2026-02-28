# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from typing import Any

import torch

from vllm.poc.core.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.v1.scheduler_params import PoCSchedulerParams


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
    assert params, "params must be non-empty"
    block_hash = params[0].block_hash
    public_key = params[0].public_key
    seq_len = params[0].seq_len
    nonces = [p.nonce for p in params]

    for p in params[1:]:
        if (
            p.block_hash != block_hash
            or p.public_key != public_key
            or p.seq_len != seq_len
        ):
            raise ValueError("PoC params group mismatch")

    return generate_inputs(
        block_hash,
        public_key,
        nonces,
        dim=hidden_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
    )


@torch.inference_mode()
def compute_poc_result(
    params: list[PoCSchedulerParams],
    *,
    last_hidden: torch.Tensor,
) -> dict[int, dict[str, Any]]:
    """Compute PoC result payloads from last-token hidden states.

    Args:
        params: PoC params for the batch, one per row of last_hidden.
        last_hidden: [batch, hidden_size] on GPU.

    Returns:
        nonce -> {"nonces": [nonce], "vectors_b64": [..]}
    """
    assert len(params) == last_hidden.shape[0]
    hidden_size = last_hidden.shape[1]

    # Normalize input (fp32) to unit sphere.
    last_hidden = last_hidden.float()
    last_hidden.div_(last_hidden.norm(dim=-1, keepdim=True).add_(1e-8))

    # Group must share block_hash/public_key/k_dim.
    block_hash = params[0].block_hash
    public_key = params[0].public_key
    k_dim = params[0].k_dim
    nonces = [p.nonce for p in params]

    for p in params[1:]:
        if p.block_hash != block_hash or p.public_key != public_key or p.k_dim != k_dim:
            raise ValueError("PoC params group mismatch")

    device = last_hidden.device
    indices = random_pick_indices(
        block_hash, public_key, nonces, hidden_size, k_dim, device
    )
    xk = torch.gather(last_hidden, 1, indices)
    yk = apply_haar_rotation(block_hash, public_key, nonces, xk, device)
    yk.div_(yk.norm(dim=-1, keepdim=True).add_(1e-8))

    vectors_f16 = yk.half().cpu().numpy()
    vectors_b64 = [
        base64.b64encode(vectors_f16[i].tobytes()).decode("ascii")
        for i in range(vectors_f16.shape[0])
    ]

    out: dict[int, dict[str, Any]] = {}
    for nonce, vec in zip(nonces, vectors_b64):
        out[nonce] = {"nonces": [nonce], "vectors_b64": [vec]}
    return out
