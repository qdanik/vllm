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
from .gpu_random import generate_householder_vector

logger = init_logger(__name__)

# Use optimized Triton kernel for layer hooks
USE_TRITON_LAYER_HOOKS = os.environ.get("POC_USE_TRITON_LAYER_HOOKS", "1") == "1"

# Track if we've logged the method being used
_method_logged = False

# Context variable for conditional hook activation
# Default False means hooks pass through unchanged (for inference)
_poc_forward_active: ContextVar[bool] = ContextVar('poc_forward_active', default=False)


@contextmanager
def poc_forward_context():
    """Context manager for PoC forward passes.
    
    Hooks only transform hidden states when this context is active.
    This allows inference and PoC to coexist without interference.
    
    Usage:
        with poc_forward_context():
            hidden_states = model(...)  # Hooks will transform
    """
    token = _poc_forward_active.set(True)
    try:
        yield
    finally:
        _poc_forward_active.reset(token)


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_forward_active.get()


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
        
        # Cache for dtype-converted vectors (lazy, per-dtype)
        self._vector_cache: dict = {}
        
        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)
            
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
        - Cached dtype conversion to avoid repeated .to() calls
        """
        apply_fn = self._apply_fn
        get_vector = lambda dtype: self._get_vector_for_dtype(layer_idx, dtype)
        
        def hook(module, input, output):
            # Early exit if not in PoC forward context - pass through unchanged
            if not is_poc_forward_active():
                return output
            
            # Get vector in correct dtype (cached)
            target_dtype = output[0].dtype if isinstance(output, tuple) else output.dtype
            v = get_vector(target_dtype)
            
            if isinstance(output, tuple):
                if len(output) >= 2:
                    # (hidden_states, residual, ...) format - transform both in-place
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    
                    # In-place transforms
                    apply_fn(hidden, v)
                    apply_fn(residual, v)
                    
                    return output  # Return same tuple (contents modified in-place)
                else:
                    # Single element tuple
                    hidden = output[0]
                    apply_fn(hidden, v)
                    return output
            else:
                apply_fn(output, v)
                return output
        
        return hook
    
    def detach(self):
        """Remove all hooks and clear caches."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []
        if hasattr(self, '_vector_cache'):
            self._vector_cache.clear()
    
    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
