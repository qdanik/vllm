"""Deterministic GPU-based random generation for PoC.

Core primitives for generating reproducible random tensors seeded by
(block_hash, public_key, nonce). Used by the production inference pipeline.
"""
import hashlib
import math
from typing import List

import torch


# Cache for seed computation - same string always gives same seed
_seed_cache: dict = {}
_SEED_CACHE_MAX = 10000  # Limit cache size


def _seed_from_string(seed_string: str) -> int:
    """Convert string to deterministic seed using SHA256.
    
    Cached to avoid repeated hashing of same strings.
    """
    cached = _seed_cache.get(seed_string)
    if cached is not None:
        return cached
    
    h = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
    seed = int(h[:8], 16)
    
    # Cache with size limit
    if len(_seed_cache) < _SEED_CACHE_MAX:
        _seed_cache[seed_string] = seed
    
    return seed


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xcc9e2d51, 0x1b873593
    
    # Work in int64 to handle uint32 range properly
    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _murmur3_32_batch(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash: keys [N], seeds [B] -> output [B, N].
    
    Computes murmur3 for all (seed, key) pairs in parallel.
    """
    c1, c2 = 0xcc9e2d51, 0x1b873593
    
    # seeds: [B], keys: [N] -> broadcast to [B, N]
    h = (seeds.unsqueeze(1) & 0xFFFFFFFF).expand(-1, keys.shape[0]).clone()
    k = (keys.to(torch.int64) & 0xFFFFFFFF).unsqueeze(0).expand(seeds.shape[0], -1)
    
    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xe6546b64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85ebca6b) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xc2b2ae35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)  # Returns int64 in [0, 2^32)
    return hashes.to(torch.float32) / 4294967296.0


def _uniform_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched uniform: seeds [B] -> output [B, n]."""
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32_batch(indices, seeds)  # [B, n]
    return hashes.to(torch.float32) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]

import os

# Try to import Triton kernel for batched uniform generation (murmur3 only)
# Can be disabled with POC_USE_TRITON_MURMUR3=0 for consensus testing
_triton_uniform_batch = None
_triton_murmur3_score_batch = None
_triton_generate_uniform_batch = None
_USE_TRITON_MURMUR3 = os.environ.get("POC_USE_TRITON_MURMUR3", "1") == "1"
try:
    from .triton_kernels import (
        triton_uniform_batch as _triton_uniform_batch_impl,
        triton_murmur3_score_batch as _triton_score_impl,
        triton_generate_uniform_batch as _triton_gen_impl,
        USE_TRITON_KERNELS,
    )
    if USE_TRITON_KERNELS and _USE_TRITON_MURMUR3:
        _triton_uniform_batch = _triton_uniform_batch_impl
        _triton_murmur3_score_batch = _triton_score_impl
        _triton_generate_uniform_batch = _triton_gen_impl
except ImportError:
    pass


def _normal_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched normal: seeds [B] -> output [B, n].
    
    Uses Triton for murmur3 (integer ops - deterministic), PyTorch for Box-Muller.
    This ensures consensus compatibility: float operations use same PyTorch code.
    """
    n_pairs = (n + 1) // 2
    
    # Try Triton for uniform generation (murmur3 only - integer ops are deterministic)
    if _triton_uniform_batch is not None:
        u = _triton_uniform_batch(seeds, n_pairs * 2, device)
        if u is not None:
            u1 = u[:, :n_pairs]
            u2 = u[:, n_pairs:]
            u1 = torch.clamp(u1, min=1e-10)
            
            # Box-Muller in PyTorch (same float ops as base image)
            r = torch.sqrt(-2.0 * torch.log(u1))
            theta = 2.0 * math.pi * u2
            z0 = r * torch.cos(theta)
            z1 = r * torch.sin(theta)
            
            result = torch.cat([z0, z1], dim=1)[:, :n]
            return result
    
    # Full PyTorch fallback
    u = _uniform_batch(seeds, n_pairs * 2, device)  # [B, n_pairs*2]
    u1 = u[:, :n_pairs]  # [B, n_pairs]
    u2 = u[:, n_pairs:]  # [B, n_pairs]
    u1 = torch.clamp(u1, min=1e-10)
    
    # Box-Muller
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = 2.0 * math.pi * u2
    z0 = r * torch.cos(theta)
    z1 = r * torch.sin(theta)
    
    # Concatenate and trim
    result = torch.cat([z0, z1], dim=1)[:, :n]  # [B, n]
    return result


def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings for PoC.
    
    Optimized version using Triton for murmur3, PyTorch for Box-Muller.
    
    Args:
        block_hash: Block hash for seeding
        public_key: Public key for seeding
        nonces: List of nonce values
        dim: Hidden dimension size
        seq_len: Sequence length
        device: Target device
        dtype: Output dtype (default float16)
    
    Returns:
        Tensor of shape [batch_size, seq_len, dim]
    """
    batch_size = len(nonces)
    elements_per_nonce = seq_len * dim
    n_pairs = (elements_per_nonce + 1) // 2
    
    # Pre-compute base seed string (shared across all nonces)
    base_seed_str = f"{block_hash}_{public_key}_nonce"
    
    # Compute all seeds on CPU then move to GPU (avoids UserWarning)
    seeds = torch.tensor(
        [_seed_from_string(f"{base_seed_str}{n}") for n in nonces],
        dtype=torch.int64
    ).to(device, non_blocking=True)
    
    # Try Triton for uniform generation (murmur3 only - integer ops are deterministic)
    if _triton_generate_uniform_batch is not None:
        u = _triton_generate_uniform_batch(seeds, n_pairs * 2, device)
        if u is not None:
            # Box-Muller in PyTorch (same float ops as base image for consensus)
            u1 = u[:, :n_pairs]
            u2 = u[:, n_pairs:]
            u1 = torch.clamp(u1, min=1e-10)
            
            r = torch.sqrt(-2.0 * torch.log(u1))
            theta = 2.0 * math.pi * u2
            z0 = r * torch.cos(theta)
            z1 = r * torch.sin(theta)
            
            result = torch.cat([z0, z1], dim=1)[:, :elements_per_nonce]
            return result.view(batch_size, seq_len, dim).to(dtype)
    
    # Fallback: use batched normal generation
    result = _normal_batch(seeds, elements_per_nonce, device)  # [batch_size, elements_per_nonce]
    
    return result.view(batch_size, seq_len, dim).to(dtype)


def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector.
    
    Args:
        block_hash: Block hash for seeding
        public_key: Public key for seeding
        dim: Target dimension
        device: Target device
        dtype: Output dtype (default float32)
    
    Returns:
        Unit vector of shape [dim]
    """
    seed_str = f"{block_hash}_{public_key}_target"
    seed = _seed_from_string(seed_str)
    normal = _normal(seed, dim, device)
    target = normal.to(dtype)
    target = target / target.norm()
    return target


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection.
    
    Args:
        seed_str: Seed string for deterministic generation
        dim: Vector dimension
        device: Target device
    
    Returns:
        Unit vector of shape [dim]
    """
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def generate_householder_vectors_batch(
    seed_strs: List[str],
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate multiple unit vectors for Householder reflections in batch.
    
    Fully vectorized version using batched murmur3 and normal generation.
    
    Args:
        seed_strs: List of seed strings
        dim: Vector dimension
        device: Target device
    
    Returns:
        Unit vectors of shape [len(seed_strs), dim]
    """
    n = len(seed_strs)
    if n == 0:
        return torch.empty(0, dim, device=device)
    
    # Pre-compute all seeds and convert to tensor (CPU then GPU)
    seeds = [_seed_from_string(s) for s in seed_strs]
    seeds_tensor = torch.tensor(seeds, dtype=torch.int64).to(device, non_blocking=True)
    
    # Generate all random vectors in one batched call
    result = _normal_batch(seeds_tensor, dim, device)
    
    # Batch normalize (in-place)
    result.div_(result.norm(dim=1, keepdim=True).clamp_(min=1e-10))
    return result


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v·x)*v
    
    Args:
        x: Input tensor of shape [..., dim]
        v: Unit vector of shape [dim] or [batch, dim]
    
    Returns:
        Transformed tensor of same shape as x
    """
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (seed-based).
    
    Vectorized: computes all nonces in parallel.
    Uses Triton kernel for murmur3 scoring when available.
    
    Args:
        block_hash: Block hash for seeding
        public_key: Public key for seeding
        nonces: List of nonce values
        dim: Full dimension size
        k: Number of dimensions to pick
        device: Target device
    
    Returns:
        Indices tensor of shape [batch_size, k] (int64)
    """
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)
    
    # Pre-compute all seeds and convert to tensor
    seeds = [_seed_from_string(f"{block_hash}_{public_key}_nonce_{n}_pick_{k}") for n in nonces]
    seeds_tensor = torch.tensor(seeds, dtype=torch.int64).to(device, non_blocking=True)
    
    # Try Triton kernel for scoring (integer ops - deterministic)
    all_scores = None
    if _triton_murmur3_score_batch is not None:
        all_scores = _triton_murmur3_score_batch(seeds_tensor, dim, device)
    
    if all_scores is None:
        # Fallback: PyTorch batched murmur3
        all_idx = torch.arange(dim, device=device, dtype=torch.int32)
        all_scores = _murmur3_32_batch(all_idx, seeds_tensor)
    
    # Batch topk: get k smallest for all nonces at once
    # Note: use -all_scores with largest=True to get smallest scores
    _, chosen = torch.topk(-all_scores, k=k, largest=True, sorted=False, dim=1)
    
    return chosen.to(torch.int64)


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: List[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections.
    
    Fully vectorized version: generates all vectors in batch, applies in batch.
    
    Args:
        block_hash: Block hash for seeding
        public_key: Public key for seeding
        nonces: List of nonce values
        x: Input vectors of shape [batch_size, k]
        device: Target device
    
    Returns:
        Rotated vectors of shape [batch_size, k]
    """
    batch_size, k = x.shape
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")
    
    y = x.clone()
    num_reflections = k - 1
    
    # Generate all seed strings at once
    seed_strs = [
        f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}"
        for nonce in nonces
        for j in range(num_reflections)
    ]
    
    # Batch generate all Householder vectors: [batch_size * (k-1), k]
    all_vectors_flat = generate_householder_vectors_batch(seed_strs, k, device)
    # Reshape to [batch_size, k-1, k]
    all_vectors = all_vectors_flat.view(batch_size, num_reflections, k).to(y.dtype)
    
    # Apply reflections in batched manner
    # For each reflection index j, apply H_j to all batch items at once
    for j in range(num_reflections):
        v = all_vectors[:, j, :]  # [batch_size, k]
        # Batched Householder: y = y - 2 * (y·v) * v
        dot = (y * v).sum(dim=-1, keepdim=True)  # [batch_size, 1]
        y = y - 2 * dot * v
    
    return y
