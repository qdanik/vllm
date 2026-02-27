# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import base64
from collections.abc import Iterable
from typing import Any

import torch

from vllm.poc.core.transforms import (
    apply_haar_rotation,
    generate_inputs,
    random_pick_indices,
)
from vllm.poc.v1.scheduler_params import PoCSchedulerParams

_EPS_F32: float = 1e-8


def _normalize_rows_f32(x: torch.Tensor, *, eps: float = _EPS_F32) -> torch.Tensor:
    """Row-wise L2 normalization in fp32.

    Args:
        x: [..., hidden_size] tensor (any floating dtype).

    Returns:
        fp32 tensor with the same shape as x.
    """
    # Work in fp32 for stability and to reduce dtype-dependent drift.
    x = x.float()
    # vector_norm is clearer than x.norm and avoids dtype surprises.
    denom = torch.linalg.vector_norm(x, ord=2, dim=-1, keepdim=True)
    denom = denom.add(eps)
    return x / denom


def _require_non_empty(params: list[PoCSchedulerParams]) -> None:
    if not params:
        raise ValueError("params must be non-empty")


def _assert_same(
    params: list[PoCSchedulerParams],
    *,
    fields: Iterable[str],
) -> None:
    """Ensure all PoCParams in the group share the given fields."""
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
        param.block_hash,
        param.public_key,
        nonces,
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
) -> dict[int, dict[str, Any]]:
    """Compute PoC result payloads from last-token hidden states.

    Args:
        params: PoC params for the batch, one per row of last_hidden.
        last_hidden: [batch, hidden_size] on GPU.

    Returns:
        nonce -> {"nonces": [nonce], "vectors_b64": [..]}
    """
    _require_non_empty(params)
    if last_hidden.ndim != 2:
        raise ValueError(
            f"last_hidden must be rank-2 [batch, hidden], got {last_hidden.shape}"
        )
    if len(params) != last_hidden.shape[0]:
        raise ValueError(
            f"params length ({len(params)}) must match batch ({last_hidden.shape[0]})"
        )

    _assert_same(params, fields=("block_hash", "public_key", "k_dim"))

    param = params[0]
    nonces = [p.nonce for p in params]

    hidden_size = int(last_hidden.shape[1])
    k_dim = int(param.k_dim)
    if not (0 < k_dim <= hidden_size):
        raise ValueError(
            f"k_dim must be in (0, hidden_size], got k_dim={k_dim}, "
            f"hidden_size={hidden_size}"
        )

    device = last_hidden.device
    last_hidden_f32 = _normalize_rows_f32(last_hidden)

    indices = random_pick_indices(
        param.block_hash,
        param.public_key,
        nonces,
        hidden_size,
        k_dim,
        device,
    )
    if indices.dtype not in (torch.int32, torch.int64):
        indices = indices.to(dtype=torch.int64)

    xk = torch.gather(last_hidden_f32, dim=1, index=indices)

    yk = apply_haar_rotation(param.block_hash, param.public_key, nonces, xk, device)
    yk = _normalize_rows_f32(yk)

    vectors_f16_cpu = yk.to(dtype=torch.float16).cpu()

    out: dict[int, dict[str, Any]] = {}
    for nonce, row in zip(nonces, vectors_f16_cpu, strict=True):
        b = row.numpy().tobytes()
        vec_b64 = base64.b64encode(b).decode("ascii")
        out[int(nonce)] = {"nonces": [int(nonce)], "vectors_b64": [vec_b64]}

    return out
