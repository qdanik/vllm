"""Per-round layer hooks for structure breaking (v2: SAFE dtype cache).

Applies deterministic Householder reflections between transformer layers
when PoC forward context is active.

Notes:
- Cache reflection vectors casted per dtype to avoid calling .to(dtype=...)
  on every forward hook invocation.
- Hook gating semantics (ContextVar)
- Reflection vectors (seed/values)
- Which outputs are transformed (hidden + residual)

⚠️ Determinism note:
The casted vectors are created deterministically from the base vectors.
We treat cached tensors as read-only.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

import torch

from vllm.poc.consensus.transforms import apply_householder, generate_householder_vector


@dataclass(frozen=True)
class PoCForwardState:
    active: bool
    apply_all: bool
    token_mask: torch.Tensor | None


_INACTIVE_STATE = PoCForwardState(active=False, apply_all=False, token_mask=None)
_poc_forward_state: ContextVar[PoCForwardState] = ContextVar(
    "poc_forward_state",
    default=_INACTIVE_STATE,
)


@contextmanager
def poc_forward_context(
    *,
    apply_all: bool = True,
    token_mask: torch.Tensor | None = None,
):
    """Activate PoC forward transforms inside this context."""
    token = _poc_forward_state.set(
        PoCForwardState(
            active=True,
            apply_all=apply_all,
            token_mask=token_mask,
        )
    )
    try:
        yield
    finally:
        _poc_forward_state.reset(token)


def get_poc_forward_state() -> PoCForwardState:
    return _poc_forward_state.get()


def is_poc_forward_active() -> bool:
    return get_poc_forward_state().active


class LayerHouseholderHook:
    """Per-round deterministic Householder transforms between transformer layers."""

    def __init__(
        self,
        model: torch.nn.Module,
        block_hash: str,
        device: torch.device,
        hidden_size: int,
    ) -> None:
        self._model = model
        self.block_hash = block_hash
        self._device = device
        self._hidden_size = hidden_size

        self._layers: list[torch.nn.Module] = []
        self._hooks: list[torch.utils.hooks.RemovableHandle] = []

        # Base vectors generated on device (dtype determined by generator).
        self._vectors: list[torch.Tensor] = []

        # v2: dtype -> list[vectors_casted]
        self._vectors_by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}

        self._attached: bool = False


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
        """Legacy setup entrypoint used by runners/tests.

        Idempotent-ish: detaches first if already attached.
        """
        if self._attached:
            self.detach()

        self._model = model
        self.block_hash = block_hash
        self._device = device
        self._hidden_size = hidden_size

        self._layers = self._find_layers(self._model)
        for i, layer in enumerate(self._layers):
            seed_str = f"{self.block_hash}_layer_{i}_householder"
            v = generate_householder_vector(seed_str, self._hidden_size, self._device)
            self._vectors.append(v)
            self._hooks.append(layer.register_forward_hook(self._make_hook(i)))

        self._attached = True

    def attach(self) -> None:
        if self._attached:
            return
        self._setup(
            model=self._model,
            block_hash=self.block_hash,
            device=self._device,
            hidden_size=self._hidden_size,
        )

    def detach(self) -> None:
        for h in self._hooks:
            h.remove()

        self._layers.clear()
        self._hooks.clear()
        self._vectors.clear()
        self._vectors_by_dtype.clear()
        self._attached = False

    @property
    def num_layers(self) -> int:
        return len(self._hooks)


    def _find_layers(self, model: torch.nn.Module) -> list[torch.nn.Module]:
        """Find transformer layers in a model-agnostic way."""
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return list(model.model.layers)
        if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return list(model.transformer.h)
        if hasattr(model, "layers"):
            return list(model.layers)
        return []

    def _get_vectors_for_dtype(self, dtype: torch.dtype) -> list[torch.Tensor]:
        """Get cached vectors cast to `dtype` (created lazily)."""
        cached = self._vectors_by_dtype.get(dtype)
        if cached is not None:
            return cached

        vecs = [v.to(dtype=dtype) for v in self._vectors]
        self._vectors_by_dtype[dtype] = vecs
        return vecs

    def _create_hook(self, layer_idx: int) -> Callable:
        """Legacy hook factory used by unit tests."""
        return self._make_hook(layer_idx)

    def _make_hook(self, layer_idx: int) -> Callable:
        def hook(module: torch.nn.Module, inputs: Any, output: Any):
            _ = module
            _ = inputs

            state = get_poc_forward_state()
            if not state.active:
                return output

            if isinstance(output, tuple):
                if len(output) == 0:
                    return output
                sample = output[0]
            else:
                sample = output

            if not isinstance(sample, torch.Tensor):
                return output

            v = self._get_vectors_for_dtype(sample.dtype)[layer_idx]

            def _apply_with_mask(x: torch.Tensor) -> torch.Tensor:
                transformed = apply_householder(x, v)
                if state.apply_all:
                    return transformed

                mask = state.token_mask
                if mask is None:
                    return x
                if mask.device != x.device:
                    mask = mask.to(device=x.device)
                if mask.shape[0] != x.shape[0]:
                    if mask.shape[0] > x.shape[0]:
                        mask = mask[:x.shape[0]]
                    else:
                        return x

                view_shape = (mask.shape[0],) + (1,) * (x.ndim - 1)
                return torch.where(mask.view(view_shape), transformed, x)

            if isinstance(output, tuple):
                if len(output) >= 2:
                    hidden, residual, *rest = output
                    hidden_out = (
                        _apply_with_mask(hidden)
                        if isinstance(hidden, torch.Tensor)
                        else hidden
                    )
                    residual_out = (
                        _apply_with_mask(residual)
                        if isinstance(residual, torch.Tensor)
                        else residual
                    )
                    return (
                        hidden_out,
                        residual_out,
                        *rest,
                    )

                if len(output) == 1:
                    (hidden,) = output
                    if isinstance(hidden, torch.Tensor):
                        return (_apply_with_mask(hidden),)
                    return output

                return output

            return _apply_with_mask(output)

        return hook
