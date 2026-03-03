"""PoC (Proof of Compute) runner plugin for the vLLM v1 GPU model runner.

A self-contained plugin that encapsulates **all** PoC-specific runtime state
and logic, following the vLLM v1 plugin pattern (cf. KVConnector, LoRA).

Lifecycle per scheduler step
----------------------------
::

    poc.begin_step(scheduler_output)              # ① batch analysis (once)
    poc.update_token_mask(total, req_idx, n)      # ② during _prepare_inputs
    poc.fill_embeds(scheduler_output)             # ③ during _preprocess
    with poc.forward_context(num_tokens_padded,   # ④ around model.forward
                             scheduler_output):
        model.forward(...)
    results = poc.extract_results(                # ⑤ in sample_tokens
                  scheduler_output, hidden)

Design principles
-----------------
* **Single source of truth** – ``has_poc`` computed once, reused everywhere.
* **Zero overhead for non-PoC batches** – early ``return`` in every method.
* **Determinism preserved** – layer hooks for all-PoC batches (proven better
  distance), in-graph Householder for mixed batches (per-token mask).
* **No runner internals leaked** – plugin accepts ``runner`` reference but
  encapsulates all PoC state in its own fields.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

# ---------------------------------------------------------------------------
# Per-step context – lightweight, computed once in ``begin_step``
# ---------------------------------------------------------------------------

class PoCStepContext:
    """Per-step PoC batch snapshot.  Cheap to create, reused throughout."""

    __slots__ = ("has_poc", "block_hash", "apply_all")

    def __init__(
        self,
        has_poc: bool = False,
        block_hash: str | None = None,
        apply_all: bool = False,
    ) -> None:
        self.has_poc = has_poc
        self.block_hash = block_hash
        self.apply_all = apply_all


_EMPTY_CTX = PoCStepContext()

# ---------------------------------------------------------------------------
# Internal: cached Householder vectors & layer hooks
# ---------------------------------------------------------------------------


class _HouseholderCache:
    """Block-hash–keyed cache for Householder reflection vectors and hooks."""

    __slots__ = ("vectors", "block_hash", "hooks")

    def __init__(self) -> None:
        self.vectors: torch.Tensor | None = None
        self.block_hash: str | None = None
        self.hooks: Any = None  # LayerHouseholderHook | None


# ---------------------------------------------------------------------------
# Plugin
# ---------------------------------------------------------------------------


class PoCRunnerPlugin:
    """PoC integration plugin for the v1 GPU model runner.

    Instantiated once; reused across steps.  All mutable PoC state lives here.
    """

    # -- construction / init -------------------------------------------------

    def __init__(self, runner: Any) -> None:
        self._runner = runner
        self._token_mask: Any = None  # CpuGpuBuffer (lazy)
        self._hh = _HouseholderCache()
        self._ctx: PoCStepContext = _EMPTY_CTX

    def initialize(self) -> None:
        """Allocate persistent GPU buffers.  Call once after runner init."""
        self._token_mask = self._runner._make_buffer(
            self._runner.max_num_tokens,
            dtype=torch.bool,
        )

    # -- public property -------------------------------------------------------

    @property
    def has_poc(self) -> bool:
        """Whether the current step contains PoC requests."""
        return self._ctx.has_poc

    # -- step 1: batch analysis ----------------------------------------------

    def begin_step(self, scheduler_output: SchedulerOutput) -> None:
        """Analyse *scheduler_output* for PoC requests.

        Must be called **once** at the start of each :meth:`execute_model`
        invocation, after :meth:`_update_states`.
        """
        poc_ids: set[str] | None = getattr(
            scheduler_output, "poc_req_ids", None
        )
        if not poc_ids:
            self._ctx = _EMPTY_CTX
            return

        block_hash = self._resolve_block_hash(scheduler_output)
        apply_all = self._check_apply_all()

        self._ctx = PoCStepContext(
            has_poc=True,
            block_hash=block_hash,
            apply_all=apply_all,
        )

    # -- step 2: token mask (called inside ``_prepare_inputs``) ---------------

    def update_token_mask(
        self,
        *,
        total_num_scheduled_tokens: int,
        req_indices: np.ndarray,
        num_reqs: int,
    ) -> None:
        """Write the per-token PoC boolean mask to GPU."""
        if not self._ctx.has_poc:
            return

        runner = self._runner
        req_ids = runner.input_batch.req_ids
        is_poc = np.fromiter(
            (
                runner.requests.get(rid) is not None
                and runner.requests[rid].poc_params is not None
                for rid in req_ids
            ),
            dtype=np.bool_,
            count=num_reqs,
        )

        mask = self._token_mask
        mask.np[:total_num_scheduled_tokens] = is_poc[req_indices]
        mask.np[total_num_scheduled_tokens:].fill(False)
        mask.copy_to_gpu(total_num_scheduled_tokens)

    # -- step 3: embedding fill (called inside ``_preprocess``) ---------------

    def fill_embeds(self, scheduler_output: SchedulerOutput) -> None:
        """Generate PoC embeddings and copy into the runner's input buffer."""
        if not self._ctx.has_poc:
            return
        from vllm.poc.v1.gpu_model_runner_integration import fill_poc_inputs_embeds

        r = self._runner
        fill_poc_inputs_embeds(
            scheduler_output,
            r.requests,
            r.input_batch,
            r.inputs_embeds.gpu,
            r.is_token_ids.gpu,
            r.device,
            r.dtype,
            r.pp_group.is_first_rank,
            r.model_config,
        )

    # -- step 4: forward context manager ------------------------------------

    @contextmanager
    def forward_context(
        self,
        *,
        num_tokens_padded: int,
        scheduler_output: SchedulerOutput,
    ) -> Generator[None, None, None]:
        """Context manager that activates PoC transforms around model forward.

        * **All-PoC batch** → layer hooks (proven better distance metric).
        * **Mixed batch** → in-graph Householder on model with per-token mask.
        * **No PoC** → no-op.
        """
        ctx = self._ctx
        if not ctx.has_poc or ctx.block_hash is None:
            yield
            return

        if ctx.apply_all:
            # Hooks path: attach hooks (cached) + activate ContextVar gate.
            with self._hooks_forward(ctx):
                yield
        else:
            # In-graph path: set Householder context on model.
            with self._in_graph_forward(
                ctx, num_tokens_padded, scheduler_output
            ):
                yield

    # -- step 5: result extraction (called inside ``sample_tokens``) ---------

    def extract_results(
        self,
        scheduler_output: SchedulerOutput,
        sample_hidden_states: torch.Tensor,
    ) -> dict[str, dict] | None:
        """Return per-request PoC computation results, or *None*."""
        if not self._ctx.has_poc:
            return None
        from vllm.poc.v1.gpu_model_runner_integration import extract_poc_results

        r = self._runner
        return extract_poc_results(
            scheduler_output,
            r.requests,
            r.input_batch,
            sample_hidden_states,
            r.device,
        )

    # =====================================================================
    # Internal helpers
    # =====================================================================

    def _resolve_block_hash(
        self, scheduler_output: SchedulerOutput
    ) -> str | None:
        block_hash: str | None = None
        requests = self._runner.requests
        for req_id in scheduler_output.num_scheduled_tokens:
            req = requests.get(req_id)
            if req is None or req.poc_params is None:
                continue
            bh = req.poc_params.block_hash
            if block_hash is None:
                block_hash = bh
            elif bh != block_hash:
                logger.warning(
                    "Batch has multiple PoC block_hash values; "
                    "using first (%s)",
                    block_hash,
                )
                break
        return block_hash

    def _check_apply_all(self) -> bool:
        runner = self._runner
        for rid in runner.input_batch.req_ids:
            req = runner.requests.get(rid)
            if req is None or req.poc_params is None:
                return False
        return True

    # -- hooks path (all-PoC) -----------------------------------------------

    @contextmanager
    def _hooks_forward(
        self, ctx: PoCStepContext
    ) -> Generator[None, None, None]:
        assert ctx.block_hash is not None
        self._ensure_hooks(ctx.block_hash)

        from vllm.poc.core.layer_hooks import poc_forward_context

        with poc_forward_context():
            yield

    def _ensure_hooks(self, block_hash: str) -> None:
        from vllm.poc.core.layer_hooks import LayerHouseholderHook

        hh = self._hh
        if (
            hh.hooks is not None
            and getattr(hh.hooks, "block_hash", None) == block_hash
            and hh.hooks.num_layers > 0
        ):
            return

        if hh.hooks is not None:
            hh.hooks.detach()

        r = self._runner
        hook = LayerHouseholderHook(
            r.get_model(), block_hash, r.device,
            r.model_config.get_hidden_size(),
        )
        hook.attach()
        hh.hooks = hook

    # -- in-graph path (mixed batches) --------------------------------------

    @contextmanager
    def _in_graph_forward(
        self,
        ctx: PoCStepContext,
        num_tokens_padded: int,
        scheduler_output: SchedulerOutput,
    ) -> Generator[None, None, None]:
        assert ctx.block_hash is not None

        target = self._find_model_target()
        if target is None:
            yield
            return

        num_layers = getattr(
            getattr(target, "config", None),
            "num_hidden_layers",
            len(getattr(target, "layers", ())),
        )
        if num_layers <= 0:
            yield
            return

        vectors = self._ensure_vectors(ctx.block_hash, num_layers)

        # Pad mask beyond scheduled tokens.
        valid = scheduler_output.total_num_scheduled_tokens
        if valid < num_tokens_padded:
            self._token_mask.gpu[valid:num_tokens_padded].fill_(False)

        try:
            target.set_poc_householder_context(
                householder_vectors=vectors,
                token_mask=self._token_mask.gpu,
                apply_all=False,
            )
            yield
        except Exception:
            logger.exception("Failed to set in-graph PoC Householder context")
            raise
        finally:
            try:
                target.clear_poc_householder_context()
            except Exception:
                logger.exception("Failed to clear in-graph PoC context")

    def _find_model_target(self) -> torch.nn.Module | None:
        runner = self._runner
        for candidate in (
            getattr(runner.get_model(), "model", None),
            runner.get_model(),
            getattr(runner.model, "model", None),
            runner.model,
        ):
            if candidate is not None and hasattr(
                candidate, "set_poc_householder_context"
            ):
                return candidate
        return None

    def _ensure_vectors(
        self, block_hash: str, num_layers: int
    ) -> torch.Tensor:
        from vllm.poc.core.transforms import generate_householder_vector

        r = self._runner
        hidden_size = r.model_config.get_hidden_size()
        dtype = r.dtype
        hh = self._hh

        if (
            hh.vectors is None
            or hh.vectors.shape != (num_layers, hidden_size)
            or hh.vectors.device != r.device
            or hh.vectors.dtype != dtype
        ):
            hh.vectors = torch.empty(
                (num_layers, hidden_size), device=r.device, dtype=dtype
            )
            hh.block_hash = None

        if hh.block_hash != block_hash:
            for i in range(num_layers):
                seed = f"{block_hash}_layer_{i}_householder"
                v = generate_householder_vector(seed, hidden_size, r.device)
                hh.vectors[i].copy_(v.to(dtype))
            hh.block_hash = block_hash

        return hh.vectors
