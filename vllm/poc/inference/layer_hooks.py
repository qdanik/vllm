"""Per-round layer hooks for structure breaking.

Applies transformations between transformer layers to break
the model's learned output structure.
"""

from contextlib import contextmanager
from contextvars import ContextVar

import torch

from vllm.poc.core.transforms import apply_householder, generate_householder_vector

# Context variable for conditional hook activation
# Default False means hooks pass through unchanged (for inference)
_poc_forward_active: ContextVar[bool] = ContextVar("poc_forward_active", default=False)

# Fast check cache - avoids ContextVar.get() overhead in hot path
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
    _poc_active_fast = True
    try:
        yield
    finally:
        _poc_active_fast = False
        _poc_forward_active.reset(token)


def is_poc_forward_active() -> bool:
    """Check if PoC forward context is active."""
    return _poc_active_fast


class LayerHouseholderHook:
    """Per-round Householder reflections applied between transformer layers.

    These hooks apply the same transform to all nonces in a round (determined
    by block_hash). Combined with per-nonce hidden state transforms, this
    provides strong structure breaking.

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
        self.hooks: list = []
        self.reflection_vectors: list[torch.Tensor] = []
        self.block_hash = block_hash
        # Cache for dtype-converted vectors
        self._vector_cache: dict[tuple[int, torch.dtype], torch.Tensor] = {}
        # self._setup(model, block_hash, device, hidden_size)

    def _find_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        # Try common patterns for different model architectures
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            # Llama, Qwen, Mistral style
            return list(model.model.layers)
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            # GPT-2 style
            return list(model.transformer.h)
        elif hasattr(model, "layers"):
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

        for i in range(len(layers)):
            seed_str = f"{block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, hidden_size, device)
            self.reflection_vectors.append(v)

            hook = layers[i].register_forward_hook(self._create_hook(i))
            self.hooks.append(hook)

    def _get_vector_for_dtype(
        self, layer_idx: int, dtype: torch.dtype
    ) -> torch.Tensor:
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
        """

        def hook(module, input, output):
            # Early exit if not in PoC forward context - pass through unchanged
            if not is_poc_forward_active():
                return output

            def transform(x):
                # Get cached vector in correct dtype
                v = self._get_vector_for_dtype(layer_idx, x.dtype)
                # Apply Householder reflection (preserves magnitude)
                return apply_householder(x, v)

            if isinstance(output, tuple):
                if len(output) >= 2:
                    # (hidden_states, residual, ...) format - transform both
                    hidden = output[0]
                    residual = output[1]
                    rest = output[2:] if len(output) > 2 else ()
                    transformed_hidden = transform(hidden)
                    transformed_residual = transform(residual)
                    return (transformed_hidden, transformed_residual) + rest
                else:
                    # Single element tuple
                    hidden = output[0]
                    transformed = transform(hidden)
                    return (transformed,)
            else:
                transformed = transform(output)
                return transformed

        return hook

    def detach(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
        self.reflection_vectors = []

    @property
    def num_layers(self) -> int:
        """Number of layers with hooks attached."""
        return len(self.hooks)
