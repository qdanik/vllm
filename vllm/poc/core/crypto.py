"""Consensus-critical deterministic RNG primitives.

Murmur3-based seeded random generation for PoC consensus.
Constants (0xcc9e2d51, 0x1b873593) and algorithms MUST NOT change.

DO NOT MODIFY without updating POC_CONSENSUS_INVARIANTS.md and golden tests.
"""

import hashlib
import math

import torch


def seed_from_string(seed_string: str) -> int:
    """Convert string seed to int32 via SHA256.

    Args:
        seed_string: Seed string (e.g. "block_hash_pubkey_nonce123")

    Returns:
        32-bit seed value
    """
    h = hashlib.sha256(seed_string.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys.

    CONSENSUS-CRITICAL: Constants 0xcc9e2d51, 0x1b873593 must not change.

    Args:
        keys: int32 tensor of any shape
        seed: 32-bit seed value

    Returns:
        int64 tensor (same shape as keys) with hash values in [0, 2^32)
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593

    # Work in int64 to handle uint32 range properly
    h = torch.full_like(keys, seed & 0xFFFFFFFF, dtype=torch.int64)
    k = keys.to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return h


def uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    """Generate uniform [0, 1) samples via Murmur3.

    Args:
        seed: 32-bit seed value
        n: Number of samples
        device: Target device

    Returns:
        float32 tensor [n] with uniform distribution
    """
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = murmur3_32(indices, seed)  # Returns int64 in [0, 2^32)
    return hashes.to(torch.float32) / 4294967296.0


def normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    """Generate standard normal samples via Box-Muller transform.

    CONSENSUS-CRITICAL: Box-Muller algorithm must not change.

    Args:
        seed: 32-bit seed value
        n: Number of samples
        device: Target device

    Returns:
        float32 tensor [n] with standard normal distribution
    """
    n_pairs = (n + 1) // 2
    u = uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


def murmur3_32_batch(keys: torch.Tensor, seeds: torch.Tensor) -> torch.Tensor:
    """Batched Murmur3 hash: keys [N], seeds [B] -> output [B, N].

    Args:
        keys: int32 tensor of shape [N]
        seeds: int64 tensor of shape [B]

    Returns:
        int64 tensor of shape [B, N] with hash values in [0, 2^32)
    """
    c1, c2 = 0xCC9E2D51, 0x1B873593
    B = seeds.shape[0]
    N = keys.shape[0]

    # Expand: h [B, N], k [B, N]
    h = seeds.view(B, 1).expand(B, N).to(torch.int64) & 0xFFFFFFFF
    k = keys.view(1, N).expand(B, N).to(torch.int64) & 0xFFFFFFFF

    k = (k * c1) & 0xFFFFFFFF
    k = ((k << 15) | (k >> 17)) & 0xFFFFFFFF
    k = (k * c2) & 0xFFFFFFFF

    h = h ^ k
    h = ((h << 13) | (h >> 19)) & 0xFFFFFFFF
    h = (h * 5 + 0xE6546B64) & 0xFFFFFFFF

    h = h ^ 4
    h = h ^ (h >> 16)
    h = (h * 0x85EBCA6B) & 0xFFFFFFFF
    h = h ^ (h >> 13)
    h = (h * 0xC2B2AE35) & 0xFFFFFFFF
    h = h ^ (h >> 16)

    return h


def uniform_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched uniform generation: seeds [B] -> output [B, n].

    Args:
        seeds: int64 tensor of shape [B]
        n: number of uniform samples per seed
        device: target device

    Returns:
        float32 tensor of shape [B, n] with values in [0, 1)
    """
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = murmur3_32_batch(indices, seeds)  # [B, n]
    return hashes.to(torch.float32) / 4294967296.0


def normal_batch(seeds: torch.Tensor, n: int, device: torch.device) -> torch.Tensor:
    """Batched normal distribution generation: seeds [B] -> output [B, n].

    Uses Box-Muller transform for consensus-safe float operations.
    Ensures exact reproducibility across runs.

    Args:
        seeds: int64 tensor of shape [B]
        n: number of normal samples per seed
        device: target device

    Returns:
        float32 tensor of shape [B, n] with standard normal distribution
    """
    n_pairs = (n + 1) // 2

    # Generate uniform samples
    u = uniform_batch(seeds, n_pairs * 2, device)  # [B, 2*n_pairs]

    u1 = u[:, :n_pairs]  # [B, n_pairs]
    u2 = u[:, n_pairs:]  # [B, n_pairs]

    # Box-Muller transform
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)

    result = torch.cat([z0, z1], dim=1)  # [B, 2*n_pairs]
    return result[:, :n]  # [B, n]
