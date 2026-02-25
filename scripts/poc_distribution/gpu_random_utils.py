"""Analysis utilities for PoC distribution experiments.

This module contains convenience wrappers for offline analysis and experiments.
These functions are NOT used in the production inference pipeline.

For production code, use the core functions from vllm.poc.core.crypto and vllm.poc.core.transforms.
"""
import hashlib
import math

import torch

# -----------------------------------------------------------------------------
# Internal helpers (duplicated from core for self-contained analysis scripts)
# -----------------------------------------------------------------------------

def _seed_from_string(seed_string: str) -> int:
    h = hashlib.sha256(seed_string.encode('utf-8')).hexdigest()
    return int(h[:8], 16)


def _murmur3_32(keys: torch.Tensor, seed: int) -> torch.Tensor:
    """Murmur3 hash for int32 keys. Returns int64 to preserve full uint32 range."""
    c1, c2 = 0xcc9e2d51, 0x1b873593
    
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


def _uniform(seed: int, n: int, device: torch.device) -> torch.Tensor:
    indices = torch.arange(n, device=device, dtype=torch.int32)
    hashes = _murmur3_32(indices, seed)
    return hashes.to(torch.float32) / 4294967296.0


def _normal(seed: int, n: int, device: torch.device) -> torch.Tensor:
    n_pairs = (n + 1) // 2
    u = _uniform(seed, n_pairs * 2, device)
    u1, u2 = u[:n_pairs], u[n_pairs:]
    u1 = torch.clamp(u1, min=1e-10)
    z0 = torch.sqrt(-2.0 * torch.log(u1)) * torch.cos(2.0 * math.pi * u2)
    z1 = torch.sqrt(-2.0 * torch.log(u1)) * torch.sin(2.0 * math.pi * u2)
    return torch.cat([z0, z1])[:n]


# -----------------------------------------------------------------------------
# Core functions (duplicated for self-contained analysis)
# -----------------------------------------------------------------------------

def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector."""
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
    """Generate a single unit vector for Householder reflection."""
    seed = _seed_from_string(seed_str)
    v = _normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v·x)*v"""
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2 * dot * v


def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (seed-based)."""
    if k <= 0 or k > dim:
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch_size = len(nonces)
    out = torch.empty(batch_size, k, device=device, dtype=torch.int64)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32)

    for i, nonce in enumerate(nonces):
        seed = _seed_from_string(
            f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}"
        )
        scores = _murmur3_32(all_idx, seed)
        _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False)
        out[i] = chosen.to(torch.int64)

    return out


def generate_haar_orthogonal_matrices(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    k: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate per-nonce Haar-random orthogonal matrices of shape [k, k]."""
    if k <= 0:
        raise ValueError(f"k must be positive, got k={k}")

    Qs = []
    for nonce in nonces:
        seed = _seed_from_string(
            f"{block_hash}_{public_key}_nonce_{nonce}_haar_qr_{k}"
        )
        A = _normal(seed, k * k, device).view(k, k).to(dtype)
        Q, R = torch.linalg.qr(A, mode="reduced")
        diag = torch.diagonal(R, 0)
        signs = torch.sign(diag)
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        Q = Q * signs.unsqueeze(0)
        Qs.append(Q)

    return torch.stack(Qs, dim=0)


# -----------------------------------------------------------------------------
# Analysis-only functions (not in production pipeline)
# -----------------------------------------------------------------------------

def generate_sign_flips(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    device: torch.device,
    round_idx: int = 0,
) -> torch.Tensor:
    """Generate per-nonce random sign flips (+1 or -1) for each dimension.
    
    This breaks hidden state clustering by randomly flipping the sign of each
    dimension, which decorrelates the directional structure while preserving
    the magnitude structure after normalization.
    """
    batch_size = len(nonces)
    signs = torch.empty(batch_size, dim, device=device, dtype=torch.float32)
    
    for i, nonce in enumerate(nonces):
        if int(round_idx) == 0:
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_signs"
        else:
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_signs_r{int(round_idx)}"
        seed = _seed_from_string(seed_str)
        u = _uniform(seed, dim, device)
        signs[i] = (u > 0.5).float() * 2 - 1
    
    return signs


def generate_nonce_transform_vectors(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    device: torch.device,
    num_reflections: int = 4,
) -> torch.Tensor:
    """Generate per-nonce Householder vectors for hidden state transform."""
    batch_size = len(nonces)
    vectors = torch.empty(batch_size, num_reflections, dim, device=device, dtype=torch.float32)
    
    for i, nonce in enumerate(nonces):
        for r in range(num_reflections):
            seed_str = f"{block_hash}_{public_key}_nonce_{nonce}_hidden_r{r}"
            vectors[i, r] = generate_householder_vector(seed_str, dim, device)
    
    return vectors


# -----------------------------------------------------------------------------
# Convenience wrappers for offline analysis
# -----------------------------------------------------------------------------

def unit_normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize vectors to unit norm along the last dimension."""
    return x / (x.norm(dim=-1, keepdim=True) + eps)


def apply_sign_flips_then_normalize(
    x: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    num_rounds: int = 1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply sign flips then normalize."""
    if x.dim() != 2:
        raise ValueError(f"x must be [B,d], got shape={tuple(x.shape)}")
    if len(nonces) != x.shape[0]:
        raise ValueError(f"len(nonces) must match batch size, got {len(nonces)} vs {x.shape[0]}")
    rounds = int(num_rounds)
    if rounds <= 0:
        raise ValueError(f"num_rounds must be positive, got {num_rounds}")
    signs = None
    for r in range(rounds):
        sr = generate_sign_flips(block_hash, public_key, nonces, x.shape[1], x.device, round_idx=r)
        signs = sr if signs is None else (signs * sr)
    x = x * signs.to(dtype=x.dtype)
    return unit_normalize(x, eps=eps)


def apply_householder_reflections(
    x_unit: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    num_reflections: int = 8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Apply per-nonce Householder reflections (full-d) and re-normalize."""
    if x_unit.dim() != 2:
        raise ValueError(f"x_unit must be [B,d], got shape={tuple(x_unit.shape)}")
    if len(nonces) != x_unit.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {x_unit.shape[0]}"
        )
    dim = x_unit.shape[1]
    tv = generate_nonce_transform_vectors(
        block_hash, public_key, nonces, dim, x_unit.device, num_reflections=num_reflections
    )
    y = x_unit
    for r in range(tv.shape[1]):
        y = apply_householder(y, tv[:, r, :].to(dtype=y.dtype))
    return unit_normalize(y, eps=eps)


def haar_rotate_k(
    xk_unit: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    nonces: list[int],
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rotate already-sliced k-vectors by per-nonce Haar Q[k,k] and re-normalize."""
    if xk_unit.dim() != 2:
        raise ValueError(f"xk_unit must be [B,k], got shape={tuple(xk_unit.shape)}")
    if len(nonces) != xk_unit.shape[0]:
        raise ValueError(
            f"len(nonces) must match batch size, got {len(nonces)} vs {xk_unit.shape[0]}"
        )
    k = int(xk_unit.shape[1])
    Q = generate_haar_orthogonal_matrices(
        block_hash, public_key, nonces, k, xk_unit.device, dtype=xk_unit.dtype
    )
    y = torch.bmm(Q, xk_unit.unsqueeze(-1)).squeeze(-1)
    return unit_normalize(y, eps=eps)


def fixed_project_full_to_k(
    x_unit_full: torch.Tensor,
    *,
    block_hash: str,
    public_key: str,
    k: int,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Project full-d unit vectors to k dims using a single fixed seeded Gaussian matrix.

    This is a *shared* (non-per-nonce) projection intended for offline analysis:
      A[d,k] ~ N(0,1) seeded by (block_hash, public_key, d, k)
      xk = normalize(x_full @ A)
    """
    if x_unit_full.dim() != 2:
        raise ValueError(f"x_unit_full must be [B,d], got shape={tuple(x_unit_full.shape)}")
    d = int(x_unit_full.shape[1])
    if k <= 0 or k > d:
        raise ValueError(f"k must be in [1, d], got k={k}, d={d}")

    x_unit_full = unit_normalize(x_unit_full, eps=eps)

    seed_str = f"{block_hash}_{public_key}_fixedproj_gaussian_raw_d{d}_k{k}"
    seed = _seed_from_string(seed_str)
    A = _normal(seed, d * int(k), x_unit_full.device).view(d, int(k)).to(dtype=x_unit_full.dtype)
    xk = x_unit_full @ A
    return unit_normalize(xk, eps=eps)

