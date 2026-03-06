"""Shared PoC in-graph Householder mixin for transformer model classes.

Provides:
- ``PoCHouseholderContext`` dataclass (per-step vectors + mask).
- ``PoCHouseholderMixin`` with set/clear helpers.
- ``poc_householder_prepare`` / ``poc_householder_apply_layer`` free functions
  used inside ``forward()`` to avoid duplicating the logic across models.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


# ---------------------------------------------------------------------------
# Context (shared dataclass, replaces per-model _PoCHouseholderContext)
# ---------------------------------------------------------------------------

@dataclass
class PoCHouseholderContext:
    """Per-step context for in-graph Householder reflections (PoC)."""
    householder_vectors: torch.Tensor | None = None
    token_mask: torch.Tensor | None = None
    apply_all: bool = False


_EMPTY_CTX = PoCHouseholderContext()


# ---------------------------------------------------------------------------
# Mixin (adds set/clear methods; expects ``self._poc_context``)
# ---------------------------------------------------------------------------

class PoCHouseholderMixin:
    """Mixin providing ``set/clear_poc_householder_context`` helpers.

    The host class **must** call ``self._init_poc_context()`` at the end of
    its ``__init__``.
    """

    _poc_context: PoCHouseholderContext

    def _init_poc_context(self) -> None:  # noqa: D401
        """Initialise PoC context attribute.  Call once in ``__init__``."""
        self._poc_context = _EMPTY_CTX

    def set_poc_householder_context(
        self,
        *,
        householder_vectors: torch.Tensor,
        token_mask: torch.Tensor | None,
        apply_all: bool = False,
    ) -> None:
        self._poc_context = PoCHouseholderContext(
            householder_vectors=householder_vectors,
            token_mask=token_mask,
            apply_all=apply_all,
        )

    def clear_poc_householder_context(self) -> None:
        self._poc_context = _EMPTY_CTX


# ---------------------------------------------------------------------------
# Forward helpers (called from model.forward to keep models DRY)
# ---------------------------------------------------------------------------

class _PoCForwardState:
    """Lightweight carrier produced by ``poc_householder_prepare``."""
    __slots__ = ("apply_fn", "vectors_cast", "mask_broadcast", "apply_all")

    def __init__(
        self,
        apply_fn,
        vectors_cast: torch.Tensor | None,
        mask_broadcast: torch.Tensor | None,
        apply_all: bool,
    ) -> None:
        self.apply_fn = apply_fn
        self.vectors_cast = vectors_cast
        self.mask_broadcast = mask_broadcast
        self.apply_all = apply_all

    @property
    def active(self) -> bool:
        return self.apply_fn is not None


_INACTIVE = _PoCForwardState(None, None, None, False)


def poc_householder_prepare(
    poc_context: PoCHouseholderContext,
    hidden_states: torch.Tensor,
) -> _PoCForwardState:
    """Prepare PoC Householder state before the layer loop.

    Returns an opaque state object; pass it to
    :func:`poc_householder_apply_layer` inside the loop.
    """
    poc_vectors = poc_context.householder_vectors
    poc_mask = poc_context.token_mask
    poc_apply_all = poc_context.apply_all

    if poc_vectors is None or (not poc_apply_all and poc_mask is None):
        return _INACTIVE

    from vllm.poc.consensus.transforms import (
        apply_householder as _apply_householder,
    )

    if poc_vectors.dtype != hidden_states.dtype:
        vectors_cast = poc_vectors.to(hidden_states.dtype)
    else:
        vectors_cast = poc_vectors

    mask_broadcast: torch.Tensor | None = None
    if not poc_apply_all:
        assert poc_mask is not None
        mask = poc_mask
        if mask.shape[0] != hidden_states.shape[0]:
            mask = mask[: hidden_states.shape[0]]
        mask_broadcast = mask.unsqueeze(-1)

    return _PoCForwardState(
        apply_fn=_apply_householder,
        vectors_cast=vectors_cast,
        mask_broadcast=mask_broadcast,
        apply_all=poc_apply_all,
    )


def poc_householder_apply_layer(
    state: _PoCForwardState,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Apply Householder reflection for a single layer.

    No-ops when *state* is inactive or *layer_idx* is out of range.
    Returns ``(hidden_states, residual)`` — possibly transformed.
    """
    if not state.active:
        return hidden_states, residual

    assert state.vectors_cast is not None
    if layer_idx >= state.vectors_cast.shape[0]:
        return hidden_states, residual

    v = state.vectors_cast[layer_idx]
    _apply = state.apply_fn

    if state.apply_all:
        hidden_states = _apply(hidden_states, v)
        if residual is not None:
            residual = _apply(residual, v)
    else:
        assert state.mask_broadcast is not None
        hidden_t = _apply(hidden_states, v)
        hidden_states = torch.where(state.mask_broadcast, hidden_t, hidden_states)
        if residual is not None:
            residual_t = _apply(residual, v)
            residual = torch.where(state.mask_broadcast, residual_t, residual)

    return hidden_states, residual
