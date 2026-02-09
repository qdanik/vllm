"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar
from typing import List

import torch

from vllm.logger import init_logger
from .gpu_random import generate_householder_vector, generate_householder_vectors_batch

logger = init_logger(__name__)

# Use optimized Triton kernel for layer hooks
USE_TRITON_LAYER_HOOKS = os.environ.get("POC_USE_TRITON_LAYER_HOOKS", "1") == "1"

# Hook mode: "hook" (register_forward_hook) or "wrap" (wrap layer.forward)
# "wrap" avoids Python forward hooks and is more compatible with compilation
POC_LAYER_HOOK_MODE = os.environ.get("POC_LAYER_HOOK_MODE", "hook").lower()
logger.info("PoC layer hook mode: %s", POC_LAYER_HOOK_MODE)

# Track if we've logged the method being used
_method_logged = False

# Context variable for conditional hook activation
# Default False means hooks pass through unchanged (for inference)
_poc_forward_active: ContextVar[bool] = ContextVar('poc_forward_active', default=False)

# Fast check cache - avoids ContextVar.get() overhead in hot path
# Set to True when entering poc_forward_context, False when exiting
_poc_active_fast: bool = False


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes.
    
    Hooks only transform hidden states when this context is active.
    This allows inference and PoC to coexist without interference.
    
    Usage:
        with poc_forward_context():
            hidden_states = model(...)  # Hooks will transform
    """
    global _poc_active_fast
    token = _poc_forward_active.set(True)
    _poc_active_fast = True  # Fast path for hooks
    try:
        yield
    finally:
        _poc_active_fast = False
        _poc_forward_active.reset(token)


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_active_fast  # Use fast path instead of ContextVar.get()


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.
    
    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.
    
    Optimizations:
    - Triton kernel for Householder reflection (when available)
    - In-place operations to minimize memory allocations
    - Pre-stacked vectors for better memory access patterns
    
    Usage:
        # At round init
        hooks = LayerHouseholderHook(model, block_hash, device, hidden_size)
        
        # Run forward passes...
        
        # At round end
        hooks.detach()
    """
    
    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        self.hooks: List = []
        self.reflection_vectors: List[torch.Tensor] = []
        self.block_hash = block_hash
        self.device = device
        self.hidden_size = hidden_size
        
        # Import optimized apply function
        self._apply_fn = self._get_apply_function()
        # self._setup(model, block_hash, device, hidden_size)
    
    def _get_apply_function(self):
        """Get the best available Householder apply function."""
        global _method_logged
        
        if USE_TRITON_LAYER_HOOKS:
            try:
                from .triton_kernels import apply_householder_triton_inplace, USE_TRITON_KERNELS
                if USE_TRITON_KERNELS:
                    if not _method_logged:
                        logger.info("Layer hooks: Using Triton kernel for Householder reflection ✓")
                        _method_logged = True
                    return apply_householder_triton_inplace
            except ImportError as e:
                if not _method_logged:
                    logger.warning(f"Layer hooks: Triton import failed ({e}), using PyTorch in-place")
                    _method_logged = True
        
        # Fallback to optimized PyTorch in-place
        if not _method_logged:
            logger.info("Layer hooks: Using PyTorch in-place for Householder reflection")
            _method_logged = True
        return self._apply_householder_inplace_pytorch
    
    @staticmethod
    def _apply_householder_inplace_pytorch(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        """Optimized in-place Householder using PyTorch."""
        # Compute dot product
        dot = (x * v).sum(dim=-1, keepdim=True)
        # In-place subtraction
        x.sub_(2 * dot * v)
        return x
    
    def _find_layers(self, model: torch.nn.Module) -> List[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        # Try common patterns for different model architectures
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, 'layers'):
            # Direct layers attribute
            return list(model.layers)
        return []
    
    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ):
        """Setup hooks on all transformer layers."""
        layers = self._find_layers(model)
        self.num_total_layers = len(layers)
        num_layers = len(layers)

        # Store original forwards when using wrap mode (for clean detach)
        self._original_forwards: List = []
        
        # Cache for dtype-converted vectors (lazy, per-dtype)
        self._vector_cache: dict = {}
        
        # Batch generate all vectors at once (94 vectors in one call)
        seed_strs = [f"{block_hash}_layer_{i}_householder" for i in range(num_layers)]
        all_vectors = generate_householder_vectors_batch(seed_strs, hidden_size, device)
        self.reflection_vectors = [all_vectors[i] for i in range(num_layers)]
        
        # Pre-cache common dtypes to avoid cache miss on first forward
        common_dtypes = [torch.float16, torch.bfloat16]
        for i, v in enumerate(self.reflection_vectors):
            for dtype in common_dtypes:
                self._vector_cache[(i, dtype)] = v.to(dtype)
        
        # Register hooks or wrap forwards after all vectors are ready
        for i in range(num_layers):
            if POC_LAYER_HOOK_MODE == "wrap":
                self._wrap_layer_forward(layers[i], i)
            else:
                hook = layers[i].register_forward_hook(self._create_hook(i))
                self.hooks.append(hook)
    
    def _get_vector_for_dtype(self, layer_idx: int, dtype: torch.dtype) -> torch.Tensor:
        """Get vector converted to specific dtype, with caching."""
        cache_key = (layer_idx, dtype)
        if cache_key not in self._vector_cache:
            self._vector_cache[cache_key] = self.reflection_vectors[layer_idx].to(dtype)
        return self._vector_cache[cache_key]
    
    def _create_hook(self, layer_idx: int):
        """Create a forward hook that applies Householder reflection.
        
        Hook only transforms when poc_forward_context is active.
        This allows inference to proceed unaffected when PoC hooks are registered.
        
        vLLM decoder layers typically return (hidden_states, residual).
        We must transform BOTH to prevent residual connections from
        preserving untransformed values.
        
        Optimizations:
        - Uses Triton kernel when available (2-3x faster)
        - In-place operations to minimize memory allocations
        - Direct vector reference (no dict lookup in hot path)
        - Fast bool check instead of ContextVar.get()
        - Pre-cached dtype conversion
        - Cached vector after first lookup per dtype
        """
        # Capture references directly - avoids dict/method lookup in hot path
        apply_fn = self._apply_fn
        vector_cache = self._vector_cache
        base_vector = self.reflection_vectors[layer_idx]
        
        # Per-hook cached vector and dtype (avoids dict lookup after first call)
        cached_v = [None]  # Use list for nonlocal mutation
        cached_dtype = [None]
        
        def hook(module, input, output):
            # Fast check - simple bool instead of ContextVar.get()
            if not _poc_active_fast:
                return output
            
            # vLLM decoder layers always return tuple (hidden_states, residual)
            hidden_states = output[0]
            target_dtype = hidden_states.dtype
            
            # Fast path: reuse cached vector if dtype matches
            v = cached_v[0]
            if v is None or cached_dtype[0] != target_dtype:
                # Cache miss - lookup or compute
                cache_key = (layer_idx, target_dtype)
                v = vector_cache.get(cache_key)
                if v is None:
                    v = base_vector.to(target_dtype)
                    vector_cache[cache_key] = v
                # Store for next call
                cached_v[0] = v
                cached_dtype[0] = target_dtype
            
            # Transform both hidden_states and residual in-place
            apply_fn(hidden_states, v)
            apply_fn(output[1], v)
            
            return output
        
        return hook

    def _wrap_layer_forward(self, layer: torch.nn.Module, layer_idx: int) -> None:
        """Wrap a layer's forward to apply Householder without forward hooks.

        This avoids Python hooks (better for torch.compile/CUDA graphs) while
        preserving exact math. The wrap is gated by poc_forward_context.
        """
        # Capture references directly - avoids dict/method lookup in hot path
        apply_fn = self._apply_fn
        vector_cache = self._vector_cache
        base_vector = self.reflection_vectors[layer_idx]

        # Per-layer cached vector and dtype (avoids dict lookup after first call)
        cached_v = [None]
        cached_dtype = [None]

        original_forward = layer.forward

        def wrapped_forward(*args, **kwargs):
            output = original_forward(*args, **kwargs)

            # Fast check - simple bool instead of ContextVar.get()
            if not _poc_active_fast:
                return output

            # vLLM decoder layers return tuple (hidden_states, residual)
            hidden_states = output[0]
            target_dtype = hidden_states.dtype

            # Fast path: reuse cached vector if dtype matches
            v = cached_v[0]
            if v is None or cached_dtype[0] != target_dtype:
                cache_key = (layer_idx, target_dtype)
                v = vector_cache.get(cache_key)
                if v is None:
                    v = base_vector.to(target_dtype)
                    vector_cache[cache_key] = v
                cached_v[0] = v
                cached_dtype[0] = target_dtype

            # Transform both hidden_states and residual in-place
            apply_fn(hidden_states, v)
            apply_fn(output[1], v)

            return output

        # Save original forward for detach
        self._original_forwards.append((layer, original_forward))
        layer.forward = wrapped_forward
    
    def detach(self):
        """Remove all hooks and clear caches."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        # Restore original forwards if wrapped
        if hasattr(self, '_original_forwards'):
            for layer, original_forward in self._original_forwards:
                layer.forward = original_forward
            self._original_forwards = []
        self.reflection_vectors = []
        if hasattr(self, '_vector_cache'):
            self._vector_cache.clear()
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
