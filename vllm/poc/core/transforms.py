"""Consensus-critical geometric transformations.

Householder reflections, Haar rotations, input/target generation.

⚠️ CONSENSUS-CRITICAL ⚠️
Algorithms MUST NOT change without updating golden tests.

This file is intentionally explicit and boring:
- deterministic seeding
- explicit shapes/dtypes
- clear invariants

Determinism note:
- We avoid torch RNG; all randomness is derived from seed_from_string + normal.
- GPU reductions (norm/sum) can be non-bitwise-deterministic across different
  hardware/drivers unless global deterministic settings are enforced.
"""

from __future__ import annotations

import torch

from vllm.poc.core.crypto import murmur3_32, normal, seed_from_string

_EPS: float = 1e-8


def _require_positive(name: str, value: int) -> None:
    if int(value) <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")


def _normalize(x: torch.Tensor, *, eps: float = _EPS) -> torch.Tensor:
    """L2-normalize on the last dim in fp32, then cast back to x.dtype."""
    x_f32 = x.float()
    denom = torch.linalg.vector_norm(x_f32, ord=2, dim=-1, keepdim=True).add(eps)
    y = x_f32 / denom
    return y.to(dtype=x.dtype)


@torch.inference_mode()
def generate_inputs(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Generate deterministic input embeddings for PoC.

    Each (block_hash, public_key, nonce) deterministically produces the same
    Tensor[seq_len, dim].

    Returns:
        Tensor[batch, seq_len, dim] on `device` with `dtype`.
    """
    _require_positive("dim", dim)
    _require_positive("seq_len", seq_len)

    batch = len(nonces)
    out = torch.empty(batch, int(seq_len), int(dim), device=device, dtype=dtype)

    # Per-nonce seeding is consensus-critical; keep the loop explicit.
    for i, nonce in enumerate(nonces):
        seed = seed_from_string(f"{block_hash}_{public_key}_nonce{int(nonce)}")
        samples = normal(seed, int(seq_len) * int(dim), device)
        out[i] = samples.view(int(seq_len), int(dim)).to(dtype)

    return out


@torch.inference_mode()
def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector.

    Each (block_hash, public_key) deterministically produces the same
    normalized Tensor[dim].
    """
    _require_positive("dim", dim)

    seed = seed_from_string(f"{block_hash}_{public_key}_target")
    v = normal(seed, int(dim), device).to(dtype)
    return _normalize(v)


@torch.inference_mode()
def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single deterministic unit vector for Householder reflection."""
    _require_positive("dim", dim)

    seed = seed_from_string(seed_str)
    v = normal(seed, int(dim), device)
    return _normalize(v)


@torch.inference_mode()
def apply_householder(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v·x)*v."""
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - (2 * dot) * v


@torch.inference_mode()
def random_pick_indices(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (seed-based).

    Implementation detail:
    - Score each dimension by a seeded hash.
    - Pick k smallest scores.
    - Deterministic tie-break by index to avoid rare hash-collision ambiguity.

    Returns:
        Tensor[batch, k] int64 on `device`.
    """
    _require_positive("dim", dim)
    _require_positive("k", k)
    if int(k) > int(dim):
        raise ValueError(f"k must be in [1, dim], got k={k}, dim={dim}")

    batch = len(nonces)
    out = torch.empty(batch, int(k), device=device, dtype=torch.int64)

    # Keep indices in int64 to form a stable composite key.
    idx_i64 = torch.arange(int(dim), device=device, dtype=torch.int64)

    for i, nonce in enumerate(nonces):
        seed = seed_from_string(
            f"{block_hash}_{public_key}_nonce_{int(nonce)}_pick_{int(k)}"
        )

        # murmur expects int32 inputs in current implementation.
        scores_i64 = murmur3_32(idx_i64.to(torch.int32), seed).to(torch.int64)

        # Tie-break deterministically by index: key = (score, idx).
        key = scores_i64 * int(dim) + idx_i64

        # Select k smallest keys.
        _, chosen = torch.topk(key, k=int(k), largest=False, sorted=False)
        out[i] = chosen

    return out


@torch.inference_mode()
def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections.

    Avoids cuSOLVER dependency (no QR decomposition). Each nonce gets a
    deterministic chain of k-1 reflections.

    Args:
        x: Tensor[batch, k]

    Returns:
        Tensor[batch, k]
    """
    if x.ndim != 2:
        raise ValueError(f"x must be rank-2 [batch, k], got shape={x.shape}")

    batch, k = x.shape
    _require_positive("k", int(k))
    if len(nonces) != batch:
        raise ValueError(f"nonces length must match batch: {len(nonces)} != {batch}")

    y = x.clone()

    for i, nonce in enumerate(nonces):
        for j in range(int(k) - 1):
            v = generate_householder_vector(
                f"{block_hash}_{public_key}_nonce_{int(nonce)}_haar_hh_{int(k)}_{int(j)}",
                int(k),
                device,
            ).to(dtype=y.dtype)
            y[i] = apply_householder(y[i], v)

    return y
