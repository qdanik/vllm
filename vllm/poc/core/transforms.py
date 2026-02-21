"""Consensus-critical geometric transformations.

Householder reflections, Haar rotations, input/target generation.
Algorithms MUST NOT change without updating golden tests.

DO NOT MODIFY without updating POC_CONSENSUS_INVARIANTS.md.
"""

import torch

from vllm.poc.core.crypto import murmur3_32, normal, seed_from_string


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

    CONSENSUS-CRITICAL: Each (block_hash, public_key, nonce) must deterministically
    produce the same [seq_len, dim] input.

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
    result = torch.empty(batch_size, seq_len, dim, device=device, dtype=dtype)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = seed_from_string(seed_str)
        normal_samples = normal(seed, seq_len * dim, device)
        result[i] = normal_samples.view(seq_len, dim).to(dtype)

    return result


def generate_target(
    block_hash: str,
    public_key: str,
    dim: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Generate deterministic target unit vector.

    CONSENSUS-CRITICAL: Each (block_hash, public_key) must deterministically
    produce the same normalized target vector.

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
    seed = seed_from_string(seed_str)
    normal_samples = normal(seed, dim, device)
    target = normal_samples.to(dtype)
    target = target / target.norm()
    return target


def generate_householder_vector(
    seed_str: str,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate a single unit vector for Householder reflection.

    CONSENSUS-CRITICAL: Each seed_str must produce the same unit vector.

    Args:
        seed_str: Seed string for deterministic generation
        dim: Vector dimension
        device: Target device

    Returns:
        Unit vector of shape [dim]
    """
    seed = seed_from_string(seed_str)
    v = normal(seed, dim, device)
    return v / v.norm()


def apply_householder(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection: H @ x = x - 2*(v·x)*v

    Reflects x across the hyperplane orthogonal to v.

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
    nonces: list[int],
    dim: int,
    k: int,
    device: torch.device,
) -> torch.Tensor:
    """Pick k dimensions per nonce deterministically (seed-based).

    CONSENSUS-CRITICAL: Scores each dimension by seeded hash and takes k smallest.
    This yields a deterministic, per-nonce subset without replacement.

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
    out = torch.empty(batch_size, k, device=device, dtype=torch.int64)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32)

    for i, nonce in enumerate(nonces):
        seed = seed_from_string(f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}")
        scores = murmur3_32(all_idx, seed)  # int64
        # Take k smallest scores via topk on the negated values (O(dim log k)).
        _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False)
        out[i] = chosen.to(torch.int64)

    return out


def apply_haar_rotation(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Apply Haar-random rotation via k-1 Householder reflections.

    CONSENSUS-CRITICAL: Avoids cuSOLVER dependency (no QR decomposition).
    Each nonce gets a deterministic chain of k-1 reflections.

    This provides a Haar-distributed random orthogonal matrix applied to x.

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

    for i, nonce in enumerate(nonces):
        for j in range(k - 1):
            v = generate_householder_vector(
                f"{block_hash}_{public_key}_nonce_{nonce}_haar_hh_{k}_{j}",
                k,
                device,
            )
            y[i] = apply_householder(y[i], v.to(y.dtype))

    return y
