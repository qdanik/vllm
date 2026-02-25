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

from vllm.poc.consensus.crypto import (
    murmur3_32_batch,
    normal,
    normal_batch,
    seed_from_string,
)

# Safety margin: keep this fraction of free GPU memory reserved so that
# the main model forward pass / NCCL / CUDA runtime have breathing room.
_GPU_MEM_SAFETY_FACTOR: float = 0.70  # use at most 70% of free VRAM

# Peak memory multiplier for normal_batch pipeline per sample:
#   murmur3_32_batch produces [sub_B, N] int64  (8 bytes)
#   then uniform_batch converts to [sub_B, N] float32  (4 bytes)
#   Box-Muller creates ~3 float32 intermediates
# Conservative estimate: ~24 bytes per element at peak.
_BYTES_PER_ELEMENT_PEAK: int = 24

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


def _estimate_gpu_sub_batch(
    n_elements: int,
    device: torch.device,
) -> int:
    """How many batch rows of *n_elements* columns fit in free GPU memory.

    Returns 0 when the device is not CUDA or when even a single row
    would not fit.
    """
    if device.type != "cuda" or not torch.cuda.is_available():
        return 0
    try:
        free, _ = torch.cuda.mem_get_info(device)
    except Exception:
        return 0
    usable = int(free * _GPU_MEM_SAFETY_FACTOR)
    row_bytes = n_elements * _BYTES_PER_ELEMENT_PEAK
    if row_bytes <= 0:
        return 0
    return max(usable // row_bytes, 0)


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
    n_elements = seq_len * dim  # columns per sample

    # Phase 1: SHA-256 seeds (CPU-only).
    seed_list = [
        seed_from_string(f"{block_hash}_{public_key}_nonce{n}") for n in nonces
    ]

    # Phase 2: determine how many rows we can process on GPU at once.
    max_gpu_rows = _estimate_gpu_sub_batch(n_elements, device)

    if max_gpu_rows >= batch_size:
        # ---- Fast path: entire batch fits on GPU in one shot ----
        seeds_gpu = torch.tensor(seed_list, device=device, dtype=torch.int64)
        samples = normal_batch(seeds_gpu, n_elements, device)
        del seeds_gpu
        return samples.view(batch_size, seq_len, dim).to(dtype)

    if max_gpu_rows >= 1:
        # ---- Sub-batch path: process in GPU-sized chunks, concat ----
        chunks: list[torch.Tensor] = []
        for start in range(0, batch_size, max_gpu_rows):
            end = min(start + max_gpu_rows, batch_size)
            sub_seeds = torch.tensor(
                seed_list[start:end],
                device=device,
                dtype=torch.int64,
            )
            sub_out = normal_batch(sub_seeds, n_elements, device)
            # Convert to target dtype immediately to free the fp32 buffer.
            chunks.append(sub_out.view(end - start, seq_len, dim).to(dtype))
            del sub_seeds, sub_out
        return torch.cat(chunks, dim=0)

    # No rows fit — raise a clear error instead of silently failing.
    raise torch.cuda.OutOfMemoryError(
        f"PoC generate_inputs: not enough GPU memory for even 1 row "
        f"(need ~{n_elements * _BYTES_PER_ELEMENT_PEAK // (1 << 20)} MiB, "
        f"max_gpu_rows=0). Reduce POC_BATCH_SIZE_DEFAULT or seq_len."
    )


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

    all_idx_i32, all_idx_i64 = _get_all_idx(hidden_size, device)

    # Phase 1: all CPU work (SHA256 seeds) up front.
    seeds = torch.tensor(
        [
            seed_from_string(f"{block_hash}_{public_key}_nonce_{n}_pick_{k_dim}")
            for n in nonces
        ],
        device=device,
        dtype=torch.int64,
    )

    # Phase 2: single batched GPU murmur3 — [B, hidden_size].
    scores = murmur3_32_batch(all_idx_i32, seeds)

    # Deterministic tie-break via composite key.
    key = scores * hidden_size + all_idx_i64.unsqueeze(0)

    # Pick k smallest keys (batched topk along last dim).
    _, chosen = torch.topk(key, k=k_dim, dim=-1, largest=False, sorted=False)
    return chosen


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

    # Outer loop over k-1 reflection steps (sequential: each depends
    # on the previous).  Inner nonce dimension is fully batched.
    for j in range(k - 1):
        # Phase 1: all CPU work (SHA256 seeds) for this step.
        seeds = torch.tensor(
            [
                seed_from_string(f"{block_hash}_{public_key}_nonce_{n}_haar_hh_{k}_{j}")
                for n in nonces
            ],
            device=device,
            dtype=torch.int64,
        )

        # Phase 2: batched GPU work — no CPU interleaving.
        v_batch = normal_batch(seeds, k, device)
        v_batch = v_batch / v_batch.norm(dim=-1, keepdim=True)
        v_batch = v_batch.to(y.dtype)

        # Batched Householder reflection: y <- y - 2*(y·v)*v
        dot = (y * v_batch).sum(dim=-1, keepdim=True)
        y = y - 2 * dot * v_batch

    return y
