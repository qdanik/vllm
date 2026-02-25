"""Consensus-critical deterministic RNG primitives (v2: safe perf tweaks).

Murmur3-based seeded random generation for PoC consensus.

⚠️ CONSENSUS-CRITICAL ⚠️
- Constants (0xcc9e2d51, 0x1b873593) MUST NOT change.
- Bit operations and ordering MUST NOT change.
- Box–Muller implementation MUST NOT change.

Notes:
- Cache torch.arange(int32) index vectors by (n, device) to reduce
  allocations in uniform()/uniform_batch().
- seed_from_string output
- murmur3_32 bit-level behavior
- Box–Muller math
"""

from __future__ import annotations

import hashlib
import math
from collections import OrderedDict

import torch

# 32-bit mask (uint32 domain)
_U32_MASK: int = 0xFFFFFFFF
_U32_FLOAT_DENOM: float = 4294967296.0  # 2**32


_IdxKey = tuple[int, str]  # (n, device_str)
_IDX_CACHE_MAX = 32
_idx_cache_i32: OrderedDict[_IdxKey, torch.Tensor] = OrderedDict()


def _device_key(device: torch.device) -> str:
    # torch.device is hashable but string is safer across torch versions.
    return str(device)


def _get_indices_i32(n: int, device: torch.device) -> torch.Tensor:
    """Get cached int32 indices [n] on `device` (read-only)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    key: _IdxKey = (n, _device_key(device))
    t = _idx_cache_i32.get(key)
    if t is not None:
        _idx_cache_i32.move_to_end(key)
        return t

    t = torch.arange(n, device=device, dtype=torch.int32)
    _idx_cache_i32[key] = t
    _idx_cache_i32.move_to_end(key)

    while len(_idx_cache_i32) > _IDX_CACHE_MAX:
        _idx_cache_i32.popitem(last=False)

    return t


def seed_from_string(seed_string: str) -> int:
    """Convert string seed to int32 via SHA256 (first 32 bits).

    Do not change without updating golden tests.
    """
    digest = hashlib.sha256(seed_string.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys.

    Args:
        keys: int32 tensor of any shape
        seed: 32-bit seed value

    Returns:
        int64 tensor (same shape as keys) with values in [0, 2^32)
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593

    h = torch.full_like(keys, seed & _U32_MASK, dtype=torch.int64)
    k = keys.to(torch.int64) & _U32_MASK

    k = (k * c1) & _U32_MASK
    k = ((k << 15) | (k >> 17)) & _U32_MASK
    k = (k * c2) & _U32_MASK

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & _U32_MASK
    h = (h * 5 + 0xE6546B64) & _U32_MASK

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _U32_MASK
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _U32_MASK
    h = h ^ (h >> 16)

    return h


@torch.inference_mode()
def uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    """Generate uniform samples in [0, 1)."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return torch.empty(0, device=device, dtype=torch.float32)

    indices = _get_indices_i32(n, device)
    hashes = murmur3_32(indices, seed)
    return hashes.to(torch.float32) / _U32_FLOAT_DENOM


@torch.inference_mode()
def normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    """Generate standard normal samples via Box–Muller."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")
    if n == 0:
        return torch.empty(0, device=device, dtype=torch.float32)

    n_pairs = (n + 1) // 2

    u = uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]

    u1 = torch.clamp(u1, min=1e-10)

    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)

    return torch.cat([z0, z1])[:n]


def murmur3_32_batch(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash: keys [N], seeds [B] -> output [B, N]."""
    c1, c2 = 0xCC9E2D51, 0x1B873593

    B = seeds.shape[0]
    N = keys.shape[0]

    h = seeds.view(B, 1).expand(B, N).to(torch.int64) & _U32_MASK
    k = keys.view(1, N).expand(B, N).to(torch.int64) & _U32_MASK

    k = (k * c1) & _U32_MASK
    k = ((k << 15) | (k >> 17)) & _U32_MASK
    k = (k * c2) & _U32_MASK

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & _U32_MASK
    h = (h * 5 + 0xE6546B64) & _U32_MASK

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & _U32_MASK
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & _U32_MASK
    h = h ^ (h >> 16)

    return h


@torch.inference_mode()
def uniform_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched uniform generation: seeds [B] -> output [B, n]."""
    indices = _get_indices_i32(n, device)
    hashes = murmur3_32_batch(indices, seeds)
    return hashes.to(torch.float32) / _U32_FLOAT_DENOM


@torch.inference_mode()
def normal_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched normal distribution generation: seeds [B] -> output [B, n]."""
    if n < 0:
        raise ValueError(f"n must be >= 0, got {n}")

    if n == 0:
        return torch.empty((seeds.shape[0], 0), device=device, dtype=torch.float32)

    n_pairs = (n + 1) // 2

    u = uniform_batch(seeds, n_pairs * 2, device)

    u1 = u[:, :n_pairs]
    u2 = u[:, n_pairs:]

    u1 = torch.clamp(u1, min=1e-10)

    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)

    result = torch.cat([z0, z1], dim=1)
    return result[:, :n]
