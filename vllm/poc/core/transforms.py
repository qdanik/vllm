"""Consensus-critical geometric transformations.

Householder reflections, Haar rotations, input/target generation.
Algorithms MUST NOT change without updating golden tests.

DO NOT MODIFY without updating POC_CONSENSUS_INVARIANTS.md.
"""


import torch
import torch.nn.functional as F

from vllm.poc.core.crypto import murmur3_32, normal, seed_from_string


# Cache for precomputed indices (deterministic by definition)
_indices_cache: dict[tuple[str, str, int, int, int, int], torch.Tensor] = {}
_INDICES_CACHE_MAX = 100

# Cache for Householder vectors (deterministic by definition)
_householder_cache: dict[str, torch.Tensor] = {}
_HOUSEHOLDER_CACHE_MAX = 1000

# Cache for arange tensors per device
_arange_cache: dict[tuple[int, int], torch.Tensor] = {}
_ARANGE_CACHE_MAX = 10


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
    # Generate in float32 then convert once (more efficient than per-iteration conversion)
    result_f32 = torch.empty(batch_size, seq_len, dim, device=device, dtype=torch.float32)

    for i, nonce in enumerate(nonces):
        seed_str = f"{block_hash}_{public_key}_nonce{nonce}"
        seed = seed_from_string(seed_str)
        normal_samples = normal(seed, seq_len * dim, device)
        result_f32[i] = normal_samples.view(seq_len, dim)
    
    # Single batch conversion to target dtype
    result = result_f32.to(dtype) if dtype != torch.float32 else result_f32

    return result


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
    # Check cache first
    cache_key = f"{seed_str}_{dim}_{id(device)}"
    if cache_key in _householder_cache:
        return _householder_cache[cache_key].clone()
    
    seed = seed_from_string(seed_str)
    v = normal(seed, dim, device)
    # Use torch.nn.functional.normalize for better GPU efficiency
    v = F.normalize(v, p=2, dim=-1)
    
    # Cache result if not at limit
    if len(_householder_cache) < _HOUSEHOLDER_CACHE_MAX:
        _householder_cache[cache_key] = v.clone()
    
    return v


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
    # Optimized dot product computation
    dot = (x * v).sum(dim=-1, keepdim=True)
    return x - 2.0 * dot * v


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
    
    # Fast path: if k == dim, just return all indices (no selection needed)
    if k == dim:
        out = torch.arange(dim, device=device, dtype=torch.int64)
        return out.unsqueeze(0).expand(batch_size, -1).contiguous()
    
    cache_key = (block_hash, public_key, batch_size, dim, k, id(device))
    
    # Check cache for precomputed indices
    if cache_key in _indices_cache:
        return _indices_cache[cache_key].clone()
    
    out = torch.empty(batch_size, k, device=device, dtype=torch.int64)
    
    # Cache arange tensor per device to avoid recreation
    arange_key = (dim, id(device))
    if arange_key in _arange_cache:
        all_idx = _arange_cache[arange_key]
    else:
        all_idx = torch.arange(dim, device=device, dtype=torch.int32)
        if len(_arange_cache) < _ARANGE_CACHE_MAX:
            _arange_cache[arange_key] = all_idx

    for i, nonce in enumerate(nonces):
        seed = seed_from_string(f"{block_hash}_{public_key}_nonce_{nonce}_pick_{k}")
        scores = murmur3_32(all_idx, seed)  # int64
        # Take k smallest scores via topk on the negated values (O(dim log k)).
        _, chosen = torch.topk(-scores, k=k, largest=True, sorted=False)
        out[i] = chosen.to(torch.int64)
    
    # Cache result if not at limit
    if len(_indices_cache) < _INDICES_CACHE_MAX:
        _indices_cache[cache_key] = out.clone()

    return out


def _apply_haar_rotation_inner(
    block_hash: str,
    public_key: str,
    nonces: list[int],
    x: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Inner implementation of Haar rotation (for torch.compile)."""
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
    return _apply_haar_rotation_inner(block_hash, public_key, nonces, x, device)
