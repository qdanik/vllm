"""Consensus-critical geometric transforms (v2: safe perf tweaks).

Notes:
- Cache torch.arange(dim) index tensors for random_pick_indices().
- Avoid repeated dtype/device casts inside tight loops.
- Seed format
- Householder math
- Haar rotation ordering
- Normalization logic

Golden tests should remain bit-identical.
"""

from __future__ import annotations

from collections import OrderedDict

import torch

from vllm.poc.core.crypto import murmur3_32, normal, seed_from_string

_IdxKey = tuple[int, str]
_IdxValue = tuple[torch.Tensor, torch.Tensor]  # (all_idx_i32, all_idx_i64)
_IDX_CACHE_MAX = 32
_idx_cache: OrderedDict[_IdxKey, _IdxValue] = OrderedDict()


def _device_key(device: torch.device) -> str:
    return str(device)


def _get_all_idx(dim: int, device: torch.device) -> _IdxValue:
    """Return (all_idx_i32, all_idx_i64) cached for (dim, device)."""
    key: _IdxKey = (dim, _device_key(device))
    result = _idx_cache.get(key)
    if result is not None:
        _idx_cache.move_to_end(key)
        return result

    all_idx_i32 = torch.arange(dim, device=device, dtype=torch.int32)
    all_idx_i64 = all_idx_i32.to(torch.int64)

    result = (all_idx_i32, all_idx_i64)
    _idx_cache[key] = result
    _idx_cache.move_to_end(key)

    while len(_idx_cache) > _IDX_CACHE_MAX:
        _idx_cache.popitem(last=False)

    return result


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    batch_size = len(nonces)
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = seed_from_string(seed_str)
        normal_samples = normal(seed, seq_len * dim, device)
        result[i] = normal_samples.view(seq_len, dim).to(dtype)

    return result


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    seed = seed_from_string(seed_str)
    v = normal(seed, dim, device)
    return v / v.norm()


def apply_householder(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    hidden_size: int,
    k_dim: int,
    device: torch.device,
) -> torch.Tensor:
    if k_dim <= 0 or k_dim > hidden_size:
        raise ValueError(f"k must be in [1, dim], got k={k_dim}, dim={hidden_size}")

    batch_size = len(nonces)
    out = torch.empty(batch_size, k_dim, device=device, dtype=torch.int64)

    all_idx_i32, all_idx_i64 = _get_all_idx(hidden_size, device)

    for i, nonce in enumerate(nonces):
        seed = seed_from_string(f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k_dim}")
        scores = murmur3_32(all_idx_i32, seed)  # int64

        # Deterministic tie-break via composite key
        key = scores * hidden_size + all_idx_i64
        # Pick k smallest keys.
        _, chosen = torch.topk(key, k=k_dim, largest=False, sorted=False)
        out[i] = chosen

    return out


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    _, k = x.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    y = x.clone()

    for i, nonce in enumerate(nonces):
        for j in range(k - 1):
            v = generate_householder_vector(
                f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}",
                k,
                device,
            )
            y[i] = apply_householder(y[i], v.to(y.dtype))

    return y
