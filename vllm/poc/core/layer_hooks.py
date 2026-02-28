"""Per-round layer hooks for structure breaking (consensus-aware).

Applies deterministic Householder reflections between transformer layers
when PoC forward context is active.

Design goals:
- Zero effect on normal inference.
- Deterministic per-round transforms (block_hash-derived).
- No hidden mutable global state.
- Explicit lifecycle (attach -> use -> detach).
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import torch

from vllm.poc.core.transforms import (
    apply_householder,
    generate_householder_vector,
)

_poc_forward_active: ContextVar[bool] = ContextVar("poc_forward_active", default=False)


@contextmanager
def poc_forward_context():
    """Activate PoC forward transforms inside this context."""
    token = _poc_forward_active.set(True)
    try:
        yield
    finally:
        _poc_forward_active.reset(token)


def is_poc_forward_active() -> bool:
    return _poc_forward_active.get()


class LayerHouseholderHook:
    """Per-round deterministic Householder transforms between transformer layers.

    Lifecycle:
        hook = LayerHouseholderHook(model, block_hash, device, hidden_size)
        hook.attach()
        ... forward passes inside poc_forward_context() ...
        hook.detach()
    """

    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ) -> None:
        self._model = model
        self._block_hash = block_hash
        self._device = device
        self._hidden_size = int(hidden_size)

        self._hooks: list[torch.utils.hooks.RemovableHandle] = []
        self._vectors: list[torch.Tensor] = []
        self._layers: list[torch.nn.Module] = []

        self._attached: bool = False

    @property
    def block_hash(self) -> str:
        return self._block_hash

    @property
    def reflection_vectors(self) -> list[torch.Tensor]:
        return self._vectors

    def _setup(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ) -> None:
        """Legacy setup entrypoint used by tests and runner integration."""
        if self._attached:
            self.detach()

        self._model = model
        self._block_hash = block_hash
        self._device = device
        self._hidden_size = int(hidden_size)

        layers = self._find_layers(self._model)
        self._layers = layers

        for i, layer in enumerate(layers):
            seed_str = f"{self._block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, self._hidden_size, self._device)
            self._vectors.append(v)
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

        self._attached = True

    def attach(self) -> None:
        if self._attached:
            return

        self._setup(self._model, self._block_hash, self._device, self._hidden_size)

    def detach(self) -> None:
        for handle in self._hooks:
            handle.remove()

        self._hooks.clear()
        self._vectors.clear()
        self._layers.clear()
        self._attached = False

    @property
    def num_layers(self) -> int:
        return len(self._hooks)

    def _create_hook(self, layer_idx: int) -> Callable:
        """Legacy hook factory used by unit tests."""
        return self._make_hook(layer_idx)

    def _find_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Model-agnostic transformer layer discovery."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)

        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)

        if hasattr(model, "layers"):
            return list(model.layers)

        return []

    def _make_hook(self, layer_idx: int) -> Callable:
        """Create forward hook for one layer."""

        def hook(module: torch.nn.Module, inputs: Any, output: Any):
            _ = module
            _ = inputs

            if not is_poc_forward_active():
                return output

            v = self._vectors[layer_idx]

            def transform(x: torch.Tensor) -> torch.Tensor:
                return apply_householder(x, v.to(dtype=x.dtype))

            if isinstance(output, tuple):
                if len(output) >= 2:
                    hidden, residual, *rest = output
                    hidden_t = transform(hidden)
                    residual_t = transform(residual)
                    return (hidden_t, residual_t, *rest)

                if len(output) == 1:
                    hidden = output[0]
                    return (transform(hidden),)

                # Empty tuple (rare but allowed by some module APIs).
                return output

            return transform(output)

        return hook
