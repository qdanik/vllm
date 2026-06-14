# SPDX-License-Identifier: Apache-2.0
"""Routing-capture forward hooks for the PoC discrete fingerprint.

Installs a ``forward_pre_hook`` on every ``FusedMoE`` module so that, while a
PoC forward is active, the raw ``router_logits`` flowing into each MoE layer is
captured into per-token routing decisions. This is the robust capture point
(architecture.md Section 5.1): the pre-hook sees ``router_logits`` on BOTH the
monolithic fp8 kernel path and the modular path, unlike the existing
``RoutedExpertsCapturer`` which captures ids only and is bypassed on the
monolithic path.

The pure ``router_logits -> [CapturedDecision]`` computation lives in
:mod:`vllm.poc.fingerprint.decisions`; this module is the stateful hook wrapper
that mirrors the install/detach lifecycle of
:class:`vllm.poc.layer_hooks.LayerHouseholderHook`.
"""
import torch

from vllm.logger import init_logger
from vllm.poc.fingerprint.decisions import router_logits_to_decisions
from vllm.poc.fingerprint.schema import CapturedDecision
from vllm.poc.layer_hooks import is_poc_forward_active

logger = init_logger(__name__)


def _is_fused_moe(module: torch.nn.Module) -> bool:
    """True if ``module`` is a ``FusedMoE`` layer.

    Imported lazily and matched by class name to avoid a hard import cycle and
    to tolerate the unit-test fake modules (which set ``top_k`` directly).
    """
    try:
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
    except Exception:  # pragma: no cover - import guard
        return False
    return isinstance(module, FusedMoE)


def find_fused_moe_modules(
    model: torch.nn.Module,
    is_moe=_is_fused_moe,
) -> list[tuple[str, torch.nn.Module]]:
    """Walk the module tree and return ``(name, module)`` for each FusedMoE.

    Ordered by the model's ``named_modules`` traversal, which is stable, so the
    enumeration index used as ``layer_idx`` is reproducible run-to-run.

    Args:
        model: Root module to walk.
        is_moe: Predicate selecting FusedMoE modules; overridable so tests can
            supply fake FusedMoE-like modules without the real heavy class.
    """
    return [
        (name, module)
        for name, module in model.named_modules()
        if is_moe(module)
    ]


def _read_router_logits(args, kwargs) -> torch.Tensor | None:
    """Extract ``router_logits`` (2nd positional / kwarg) from a FusedMoE call."""
    if "router_logits" in kwargs:
        return kwargs["router_logits"]
    if len(args) >= 2:
        return args[1]
    return None


class RoutingFingerprintHook:
    """Per-block_hash routing capture via FusedMoE forward_pre_hooks.

    Lifecycle mirrors :class:`LayerHouseholderHook`: construct with a
    ``block_hash``, ``_setup`` installs one pre-hook per FusedMoE module; a
    later block_hash detaches and reinstalls. Hooks no-op unless
    :func:`is_poc_forward_active` is true, so inference is never touched.
    """

    def __init__(self, block_hash: str):
        self.block_hash = block_hash
        self.hooks: list = []
        # layer_idx -> top_k captured at install time.
        self._top_k_by_layer: dict[int, int] = {}
        # (nonce_idx, position_idx, CapturedDecision) accumulated per forward.
        self._buffer: list[tuple[int, int, CapturedDecision]] = []
        self._warned_internal_router = False

    def _setup(self, model: torch.nn.Module, is_moe=_is_fused_moe) -> None:
        """Install a forward_pre_hook on every FusedMoE module in ``model``.

        ``is_moe`` is overridable so tests can target fake FusedMoE-like modules.
        """
        moe_modules = find_fused_moe_modules(model, is_moe=is_moe)
        for layer_idx, (name, module) in enumerate(moe_modules):
            top_k = int(module.top_k)
            self._top_k_by_layer[layer_idx] = top_k
            self._guard_internal_router(name, module)
            handle = module.register_forward_pre_hook(
                self._make_pre_hook(layer_idx), with_kwargs=True
            )
            self.hooks.append(handle)
        logger.info(
            "RoutingFingerprintHook: installed on %d FusedMoE modules",
            len(self.hooks),
        )

    def _guard_internal_router(self, name: str, module: torch.nn.Module) -> None:
        """Warn once if a module uses an internal router.

        For the three target models the gate is external, so the pre-hook sees
        true gate logits. An internal router computes routing inside the kernel,
        and the ``router_logits`` argument is then not the gate output we want;
        we warn so the calibration run is not silently mis-captured.
        """
        is_internal = getattr(module, "is_internal_router", False)
        if is_internal and not self._warned_internal_router:
            logger.warning(
                "RoutingFingerprintHook: FusedMoE '%s' reports "
                "is_internal_router=True; captured router_logits may not be the "
                "true gate output. Routing fingerprint may be unreliable.",
                name,
            )
            self._warned_internal_router = True

    def _make_pre_hook(self, layer_idx: int):
        """Build the pre-hook closure for one FusedMoE layer index."""
        top_k = self._top_k_by_layer[layer_idx]

        def pre_hook(module, args, kwargs):
            if not is_poc_forward_active():
                return None
            router_logits = _read_router_logits(args, kwargs)
            if router_logits is None or router_logits.dim() != 2:
                return None
            # seq_len is filled in by the caller before each forward via the
            # buffer-bound closure variable on the hook instance.
            seq_len = self._active_seq_len
            decisions = router_logits_to_decisions(
                router_logits, top_k, layer_idx, seq_len
            )
            self._buffer.extend(decisions)
            return None

        return pre_hook

    # The active per-nonce sequence length, set by the runner around a forward.
    _active_seq_len: int = 1

    def set_seq_len(self, seq_len: int) -> None:
        """Set the per-nonce sequence length used to map token rows to sites."""
        self._active_seq_len = int(seq_len)

    def drain(self) -> list[tuple[int, int, CapturedDecision]]:
        """Return and clear the buffered routing decisions.

        Returns:
            List of ``(nonce_idx, position_idx, CapturedDecision)`` captured
            since the last drain.
        """
        drained = self._buffer
        self._buffer = []
        return drained

    def detach(self) -> None:
        """Remove all installed hooks and clear state."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []
        self._top_k_by_layer = {}
        self._buffer = []

    @property
    def num_layers(self) -> int:
        """Number of FusedMoE modules currently hooked."""
        return len(self.hooks)
