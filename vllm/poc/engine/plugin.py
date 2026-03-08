"""PoC runner plugin for the vLLM v1 GPU model runner.

This plugin keeps all PoC runtime state in one place and exposes five hooks
used by the runner pipeline per scheduler step:

1) ``begin_step``
2) ``update_token_mask``
3) ``fill_embeds``
4) ``forward_context``
5) ``extract_results``

Design goals:
- No-op overhead for non-PoC batches.
- Deterministic PoC transforms for both all-PoC and mixed batches.
- Minimal coupling to runner internals.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from vllm.poc._log import init_poc_logger

if TYPE_CHECKING:
    from collections.abc import Generator

    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_poc_logger(__name__)


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


class _HouseholderCache:
    """Block-hash–keyed cache for Householder reflection vectors and hooks."""

    __slots__ = ("hooks",)

    def __init__(self) -> None:
        self.hooks: Any = None  # LayerHouseholderHook | None


class PoCRunnerPlugin:
    """PoC integration plugin for the v1 GPU model runner.

    Instantiated once; reused across steps. All mutable PoC state lives here.
    """

    # -- construction / init -------------------------------------------------

    def __init__(self, runner: Any) -> None:
        self._runner = runner
        self._token_mask: Any = None  # CpuGpuBuffer (lazy)
        self._hh = _HouseholderCache()
        self._ctx: PoCStepContext = _EMPTY_CTX

    def initialize(self) -> None:
        """Allocate persistent GPU buffers. Call once after runner init."""
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
        apply_all = self._check_apply_all(scheduler_output)

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
        from vllm.poc.engine.gpu import fill_poc_inputs_embeds

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

        * **All-PoC batch** → layer hooks over all tokens.
        * **Mixed batch** → layer hooks with per-token mask.
        * **No PoC** → no-op.
        """
        ctx = self._ctx
        if not ctx.has_poc or ctx.block_hash is None:
            yield
            return

        with self._hooks_forward(ctx, num_tokens_padded, scheduler_output):
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
        from vllm.poc.engine.gpu import extract_poc_results

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

    def _check_apply_all(self, scheduler_output: SchedulerOutput) -> bool:
        poc_ids: set[str] | None = getattr(
            scheduler_output, "poc_req_ids", None
        )
        if not poc_ids:
            return False
        num_scheduled = len(scheduler_output.num_scheduled_tokens)
        return num_scheduled > 0 and len(poc_ids) == num_scheduled

    # -- hooks path (all-PoC) -----------------------------------------------

    @contextmanager
    def _hooks_forward(
        self,
        ctx: PoCStepContext,
        num_tokens_padded: int,
        scheduler_output: SchedulerOutput,
    ) -> Generator[None, None, None]:
        assert ctx.block_hash is not None
        self._ensure_hooks(ctx.block_hash)

        from vllm.poc.consensus.hooks import poc_forward_context

        token_mask: torch.Tensor | None = None
        if not ctx.apply_all:
            # Pad mask beyond scheduled tokens.
            valid = scheduler_output.total_num_scheduled_tokens
            if valid < num_tokens_padded:
                self._token_mask.gpu[valid:num_tokens_padded].fill_(False)
            token_mask = self._token_mask.gpu

        with poc_forward_context(apply_all=ctx.apply_all, token_mask=token_mask):
            yield

    def _ensure_hooks(self, block_hash: str) -> None:
        from vllm.poc.consensus.hooks import LayerHouseholderHook

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
