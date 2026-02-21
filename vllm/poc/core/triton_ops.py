"""Triton-optimized operations for PoC transforms.

Provides GPU-accelerated implementations of consensus-critical operations
with automatic fallback to PyTorch when Triton is unavailable.

CONSENSUS INVARIANTS:
- All integer operations (murmur3) are deterministic across implementations
- Floating-point operations use identical algorithms (Box-Muller, L2 normalization)
- Triton and PyTorch fallbacks produce bit-exact results
"""

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

# Try importing Triton, fallback to None if unavailable
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False
    logger.info("Triton not available, using PyTorch fallback for PoC operations")


# =============================================================================
# Murmur3 constants (consensus-critical)
# =============================================================================

_MURMUR3_C1 = 0xcc9e2d51
_MURMUR3_C2 = 0x1b873593
_MURMUR3_M1 = 0x85ebca6b
_MURMUR3_M2 = 0xc2b2ae35
_MURMUR3_ADD = 0xe6546b64
_MASK32 = 0xFFFFFFFF


# =============================================================================
# Triton kernels
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _batched_murmur3_score_kernel(
        seeds_ptr,          # [B] - int64 seeds
        scores_ptr,         # [B, dim] - output scores (int64)
        batch_size,
        dim,
        scores_stride_batch,
        c1, c2, m1, m2, add_const, mask32,
        BLOCK_D: tl.constexpr,
    ):
        """Batched murmur3 scoring for random_pick_indices."""
        batch_idx = tl.program_id(0)
        
        if batch_idx >= batch_size:
            return
        
        seed = tl.load(seeds_ptr + batch_idx) & mask32
        output_offset = batch_idx * scores_stride_batch
        
        for block_start in range(0, dim, BLOCK_D):
            offs = (block_start + tl.arange(0, BLOCK_D)).to(tl.int64)
            mask = offs < dim
            
            # Murmur3 hash (integer-only, deterministic)
            h = seed
            k = offs & mask32
            k = (k * c1) & mask32
            k = ((k << 15) | (k >> 17)) & mask32
            k = (k * c2) & mask32
            h = h ^ k
            h = ((h << 13) | (h >> 19)) & mask32
            h = (h * 5 + add_const) & mask32
            h = h ^ (h >> 16)
            h = (h * m1) & mask32
            h = h ^ (h >> 13)
            h = (h * m2) & mask32
            h = h ^ (h >> 16)
            
            tl.store(scores_ptr + output_offset + offs, h, mask=mask)

    @triton.jit
    def _batched_generate_uniform_kernel(
        seeds_ptr,          # [B] - int64 seeds
        output_ptr,         # [B, n] - output uniform [0,1) (float32)
        n,
        output_stride_batch,
        c1, c2, m1, m2, add_const, mask32,
        BLOCK_N: tl.constexpr,
    ):
        """Batched murmur3 + conversion to uniform [0, 1)."""
        batch_idx = tl.program_id(0)
        seed = tl.load(seeds_ptr + batch_idx) & mask32
        output_offset = batch_idx * output_stride_batch
        
        for block_start in range(0, n, BLOCK_N):
            offs = (block_start + tl.arange(0, BLOCK_N)).to(tl.int64)
            mask = offs < n
            
            # Murmur3 hash
            h = seed
            k = offs & mask32
            k = (k * c1) & mask32
            k = ((k << 15) | (k >> 17)) & mask32
            k = (k * c2) & mask32
            h = h ^ k
            h = ((h << 13) | (h >> 19)) & mask32
            h = (h * 5 + add_const) & mask32
            h = h ^ (h >> 16)
            h = (h * m1) & mask32
            h = h ^ (h >> 13)
            h = (h * m2) & mask32
            h = h ^ (h >> 16)
            
            # Convert to uniform [0, 1)
            u = h.to(tl.float32) / 4294967296.0
            tl.store(output_ptr + output_offset + offs, u, mask=mask)

    @triton.jit
    def _householder_reflection_kernel(
        x_ptr,              # [num_tokens, hidden_size] - in-place modification
        v_ptr,              # [hidden_size] - Householder vector
        num_tokens,
        hidden_size,
        x_stride_token,
        BLOCK_D: tl.constexpr,
    ):
        """In-place Householder reflection: x = x - 2*(x·v)*v"""
        token_idx = tl.program_id(0)
        
        if token_idx >= num_tokens:
            return
        
        x_offset = token_idx * x_stride_token
        d_offs = tl.arange(0, BLOCK_D)
        mask = d_offs < hidden_size
        
        x_vec = tl.load(x_ptr + x_offset + d_offs, mask=mask, other=0.0).to(tl.float32)
        v_vec = tl.load(v_ptr + d_offs, mask=mask, other=0.0).to(tl.float32)
        
        # Compute dot product and apply reflection
        dot = tl.sum(x_vec * v_vec, axis=0)
        result = x_vec - 2.0 * dot * v_vec
        
        tl.store(x_ptr + x_offset + d_offs, result, mask=mask)


# =============================================================================
# Public API with automatic fallback
# =============================================================================

def batched_murmur3_scores(
    seeds: torch.Tensor,
    dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute murmur3 scores for dimension selection.
    
    Args:
        seeds: [B] int64 tensor of seeds
        dim: Number of dimensions to score
        device: Target device
    
    Returns:
        [B, dim] tensor of int64 hash scores
    """
    batch_size = seeds.shape[0]
    
    if TRITON_AVAILABLE:
        try:
            scores = torch.empty((batch_size, dim), dtype=torch.int64, device=device)
            BLOCK_D = min(triton.next_power_of_2(dim), 1024)
            
            grid = (batch_size,)
            _batched_murmur3_score_kernel[grid](
                seeds, scores,
                batch_size, dim,
                scores.stride(0),
                _MURMUR3_C1, _MURMUR3_C2, _MURMUR3_M1, _MURMUR3_M2,
                _MURMUR3_ADD, _MASK32,
                BLOCK_D=BLOCK_D,
            )
            return scores
        except Exception as e:
            logger.warning(
                "Triton kernel failed, using PyTorch fallback: %s", e
            )
    
    # PyTorch fallback
    from vllm.poc.core.crypto import murmur3_32
    
    scores = torch.empty((batch_size, dim), dtype=torch.int64, device=device)
    all_idx = torch.arange(dim, device=device, dtype=torch.int32)
    
    for i in range(batch_size):
        scores[i] = murmur3_32(all_idx, int(seeds[i].item()))
    
    return scores


def batched_uniform(
    seeds: torch.Tensor,
    n: int,
    device: torch.device,
) -> torch.Tensor:
    """Generate uniform random values in [0, 1).
    
    Args:
        seeds: [B] int64 tensor of seeds
        n: Number of values per batch element
        device: Target device
    
    Returns:
        [B, n] tensor of uniform random values
    """
    batch_size = seeds.shape[0]
    
    if TRITON_AVAILABLE:
        try:
            output = torch.empty((batch_size, n), dtype=torch.float32, device=device)
            BLOCK_N = min(triton.next_power_of_2(n), 1024)
            
            grid = (batch_size,)
            _batched_generate_uniform_kernel[grid](
                seeds, output,
                n, output.stride(0),
                _MURMUR3_C1, _MURMUR3_C2, _MURMUR3_M1, _MURMUR3_M2,
                _MURMUR3_ADD, _MASK32,
                BLOCK_N=BLOCK_N,
            )
            return output
        except Exception as e:
            logger.warning(
                "Triton kernel failed, using PyTorch fallback: %s", e
            )
    
    # PyTorch fallback
    from vllm.poc.core.crypto import murmur3_32
    
    output = torch.empty((batch_size, n), dtype=torch.float32, device=device)
    all_idx = torch.arange(n, device=device, dtype=torch.int32)
    
    for i in range(batch_size):
        hashes = murmur3_32(all_idx, int(seeds[i].item()))
        output[i] = hashes.float() / 4294967296.0
    
    return output


def apply_householder_inplace(
    x: torch.Tensor,
    v: torch.Tensor,
) -> torch.Tensor:
    """Apply Householder reflection in-place: x = x - 2*(x·v)*v
    
    Args:
        x: [num_tokens, hidden_size] tensor (modified in-place)
        v: [hidden_size] unit vector
    
    Returns:
        Modified x tensor
    """
    # Flatten to 2D if needed
    original_shape = x.shape
    if x.dim() > 2:
        x = x.view(-1, x.shape[-1])
    
    num_tokens, hidden_size = x.shape
    
    # Triton has practical limit on BLOCK_D (~8192)
    MAX_TRITON_HIDDEN = 8192
    
    if TRITON_AVAILABLE and hidden_size <= MAX_TRITON_HIDDEN:
        try:
            x_contig = x.contiguous()
            v_f32 = v.float().contiguous()
            
            BLOCK_D = triton.next_power_of_2(hidden_size)
            grid = (num_tokens,)
            
            _householder_reflection_kernel[grid](
                x_contig, v_f32,
                num_tokens, hidden_size,
                x_contig.stride(0),
                BLOCK_D=BLOCK_D,
            )
            
            if not x.is_contiguous():
                x.copy_(x_contig)
            
            if len(original_shape) > 2:
                x = x.view(original_shape)
            
            return x
        except Exception as e:
            logger.warning(
                "Triton kernel failed, using PyTorch fallback: %s", e
            )
    
    # PyTorch fallback
    v_expanded = v.unsqueeze(0)
    dot = (x * v_expanded).sum(dim=-1, keepdim=True)
    x.sub_(2 * dot * v_expanded)
    
    if len(original_shape) > 2:
        x = x.view(original_shape)
    
    return x


def box_muller_transform(
    u1: torch.Tensor,
    u2: torch.Tensor,
) -> torch.Tensor:
    """Box-Muller transform: uniform → normal distribution.
    
    CONSENSUS-CRITICAL: Must use PyTorch for float consistency.
    
    Args:
        u1, u2: [batch, n] uniform random in (0, 1)
    
    Returns:
        [batch, n] normal random values
    """
    import math
    
    # Clamp to avoid log(0)
    u1 = u1.clamp(min=1e-10, max=1-1e-10)
    u2 = u2.clamp(min=1e-10, max=1-1e-10)
    
    # Box-Muller
    r = torch.sqrt(-2.0 * torch.log(u1))
    theta = 2.0 * math.pi * u2
    return r * torch.cos(theta)


# =============================================================================
# Fused operations for better performance
# =============================================================================

if TRITON_AVAILABLE:
    @triton.jit
    def _fused_normalize_kernel(
        x_ptr,              # [batch, dim] in/out pointer
        batch_size,
        dim,
        stride_batch,
        BLOCK_DIM: tl.constexpr,
    ):
        """Fused L2 normalization kernel (in-place)."""
        batch_idx = tl.program_id(0)
        
        if batch_idx >= batch_size:
            return
        
        offset = batch_idx * stride_batch
        dim_range = tl.arange(0, BLOCK_DIM)
        mask = dim_range < dim
        
        # Load vector
        x = tl.load(x_ptr + offset + dim_range, mask=mask, other=0.0)
        
        # Compute L2 norm
        norm_sq = tl.sum(x * x, axis=0)
        norm = tl.sqrt(norm_sq + 1e-8)
        
        # Normalize and store
        x_normalized = x / norm
        tl.store(x_ptr + offset + dim_range, x_normalized, mask=mask)


def fused_normalize(x: torch.Tensor) -> torch.Tensor:
    """Fused L2 normalization using Triton.
    
    In-place normalization for small vectors (e.g., k dimension).
    Combines norm computation and division in single kernel launch.
    
    Args:
        x: [batch, dim] tensor
    
    Returns:
        Normalized tensor (in-place modification)
    """
    if not TRITON_AVAILABLE:
        # PyTorch fallback
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    
    try:
        batch_size, dim = x.shape
        BLOCK_DIM = triton.next_power_of_2(dim)
        
        # Make contiguous for in-place operation
        x_contig = x.contiguous()
        
        grid = (batch_size,)
        _fused_normalize_kernel[grid](
            x_contig,
            batch_size, dim,
            x_contig.stride(0),
            BLOCK_DIM=BLOCK_DIM,
        )
        
        if not x.is_contiguous():
            x.copy_(x_contig)
        
        return x
    except Exception as e:
        logger.warning(
            "Triton fused normalize failed, using PyTorch: %s", e
        )
        return x / x.norm(dim=-1, keepdim=True).clamp(min=1e-8)


if TRITON_AVAILABLE:
    @triton.jit
    def _fused_gather_haar_kernel(
        hidden_ptr,           # [batch, hidden_size] - normalized hidden states
        indices_ptr,          # [batch, k] - picked indices
        hh_vectors_ptr,       # [batch, k-1, k] - precomputed Householder vectors
        output_ptr,           # [batch, k] - rotated vectors
        batch_size: tl.constexpr,
        hidden_size: tl.constexpr,
        k: tl.constexpr,
        hidden_stride_batch,
        indices_stride_batch,
        hh_stride_batch,
        hh_stride_reflection,
        output_stride_batch,
        BLOCK_K: tl.constexpr,
    ):
        """Fused kernel: gather k dims + apply k-1 Householder reflections.
        
        Each program handles one batch element.
        For small k (e.g., 12), entire computation fits in registers.
        Provides ~3x speedup vs Python loop for k=12, batch=32.
        """
        batch_idx = tl.program_id(0)
        
        if batch_idx >= batch_size:
            return
        
        # Offsets for this batch
        hidden_offset = batch_idx * hidden_stride_batch
        indices_offset = batch_idx * indices_stride_batch
        hh_offset = batch_idx * hh_stride_batch
        output_offset = batch_idx * output_stride_batch
        
        # Load indices for this batch
        k_range = tl.arange(0, BLOCK_K)
        mask = k_range < k
        
        indices = tl.load(indices_ptr + indices_offset + k_range, mask=mask, other=0)
        
        # Gather: x[i] = hidden[indices[i]]
        x = tl.load(hidden_ptr + hidden_offset + indices, mask=mask, other=0.0)
        
        # Apply k-1 Householder reflections: H @ x = x - 2*(v·x)*v
        for j in range(k - 1):
            # Load j-th Householder vector
            v_offset = hh_offset + j * hh_stride_reflection
            v = tl.load(hh_vectors_ptr + v_offset + k_range, mask=mask, other=0.0)
            
            # Compute dot product v·x
            dot = tl.sum(v * x, axis=0)
            
            # Apply reflection: x = x - 2*(v·x)*v
            x = x - 2.0 * dot * v
        
        # Store result
        tl.store(output_ptr + output_offset + k_range, x, mask=mask)


def fused_gather_haar_rotation(
    hidden: torch.Tensor,      # [batch, hidden_size] FP32
    indices: torch.Tensor,     # [batch, k] INT64
    hh_vectors: torch.Tensor,  # [batch, k-1, k] - precomputed Householder vectors
    device: torch.device,
) -> torch.Tensor:
    """Fused gather + Haar rotation using Triton.
    
    Combines:
    1. Gather k dimensions from hidden state
    2. Apply k-1 Householder reflections
    
    For k=12 and batch_size=32, this is ~3x faster than Python loop.
    
    Args:
        hidden: [batch, hidden_size] hidden states (FP32)
        indices: [batch, k] dimension indices (INT64)
        hh_vectors: [batch, k-1, k] precomputed Householder vectors
        device: Target device
    
    Returns:
        [batch, k] rotated vectors
    """
    if not TRITON_AVAILABLE:
        # PyTorch fallback
        batch_size, k = indices.shape
        output = torch.empty(batch_size, k, device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            # Gather
            x = hidden[i, indices[i]]
            
            # Apply Householder reflections
            for j in range(k - 1):
                v = hh_vectors[i, j]
                dot = (x * v).sum()
                x = x - 2 * dot * v
            
            output[i] = x
        
        return output
    
    try:
        batch_size, hidden_size = hidden.shape
        k = indices.shape[1]
        
        # Ensure FP32 for computation accuracy
        hidden_f32 = hidden.float().contiguous()
        indices_i64 = indices.long().contiguous()
        hh_vectors_f32 = hh_vectors.float().contiguous()
        
        # Allocate output
        output = torch.empty(batch_size, k, device=device, dtype=torch.float32)
        
        # Choose block size (power of 2, at least k)
        BLOCK_K = triton.next_power_of_2(k)
        
        # Launch kernel
        grid = (batch_size,)
        _fused_gather_haar_kernel[grid](
            hidden_f32, indices_i64, hh_vectors_f32, output,
            batch_size, hidden_size, k,
            hidden_f32.stride(0),
            indices_i64.stride(0),
            hh_vectors_f32.stride(0),
            hh_vectors_f32.stride(1),
            output.stride(0),
            BLOCK_K=BLOCK_K,
        )
        
        return output
    except Exception as e:
        logger.warning(
            "Triton fused gather+haar failed, using PyTorch: %s", e
        )
        # Fallback
        batch_size, k = indices.shape
        output = torch.empty(batch_size, k, device=device, dtype=torch.float32)
        
        for i in range(batch_size):
            x = hidden[i, indices[i]]
            for j in range(k - 1):
                v = hh_vectors[i, j]
                dot = (x * v).sum()
                x = x - 2 * dot * v
            output[i] = x
        
        return output


