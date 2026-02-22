# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Async PoC worker state (v1).

This module encapsulates the fire-and-forget PoC execution state used by the
v1 GPU worker:

- PoC runs in background Python threads on all TP ranks (collective execution).
- Rank 0 manages event polling and result collection.
- Each rank uses its own CUDA stream for device-local GPU work.
- Completion is detected via CUDA event polling on rank 0 only.

Design rationale:
- PoC requires collective communication (TP operations in the model).
- All ranks must execute concurrently to avoid deadlock in collective ops.
- Only rank 0 manages async lifecycle (event polling, result storage).
- Each rank uses separate CUDA stream (one per device), not competing for resources.
- This ensures single logical PoC execution using all GPUs collectively.

The intent is to keep the main `Worker` class readable while ensuring:
- normal inference never blocks on PoC
- PoC failures (incl. CUDA OOM) are isolated and reported as request-level errors
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from vllm.distributed import get_tp_group
from vllm.poc.utils.poc_logger import init_poc_logger
from vllm.v1.core.sched.output import PoCRequestData

logger = init_poc_logger(__name__)


class AsyncPoCWorker:
    """Holds all mutable state for async PoC execution on a worker.

    PoC execution is collective across all TP ranks:
    - Only rank 0 creates the async thread and manages completion events.
    - All ranks participate in PoC forward (required for model TP operations).
    - Each rank uses its own CUDA stream for device-local GPU work.
    - Synchronization via TP group barrier ensures coherent execution.
    """

    def __init__(self, worker: Any) -> None:
        self.worker = worker
        self.tp_group = None  # Lazy init after distributed is set up
        self.is_tp_rank_0 = False
        self.stream: torch.cuda.Stream | None = None

        # Thread management
        self._event: torch.cuda.Event | None = None
        self._thread: threading.Thread | None = None
        self._should_abort = threading.Event()  # Flag for graceful shutdown

        self._pending_result: dict[str, Any] | None = None
        self._pending_error: BaseException | None = None
        self._pending_request_id: str | None = None

    def _ensure_distributed_init(self) -> None:
        """Lazy initialization of distributed state."""
        if self.tp_group is None:
            try:
                self.tp_group = get_tp_group()
                self.is_tp_rank_0 = self.tp_group.rank_in_group == 0
            except (AssertionError, RuntimeError, Exception) as e:
                # Distributed not yet initialized or other error; treat as single-GPU
                self.is_tp_rank_0 = True
                logger.debug(
                    "TP group not initialized or error occurred (%s), "
                    "assuming single-GPU mode",
                    type(e).__name__,
                )

    def has_in_flight(self) -> bool:
        try:
            self._ensure_distributed_init()
            if self.is_tp_rank_0:
                # Check if thread exists and is still running
                if self._thread is not None and self._thread.is_alive():
                    return True
                # Clean up dead thread
                if self._thread is not None and not self._thread.is_alive():
                    self._cleanup_thread()
            # Non-rank-0: check via barrier timeout (heuristic)
            return False  # Scheduler only queries rank 0
        except Exception as e:
            logger.warning("Error in has_in_flight: %s", e)
            return False

    def _cleanup_thread(self) -> None:
        """Clean up dead thread state."""
        if self._thread is not None:
            try:
                self._thread.join(timeout=0.1)
            except Exception:
                pass
            self._thread = None
        self._event = None
        self._should_abort.clear()

    def launch_if_idle(
        self,
        worker: Any,
        poc_req: PoCRequestData,
        *,
        hidden_size: int,
    ) -> bool:
        """Launch PoC if no previous PoC is currently in-flight.

        Returns True if a new PoC was launched, False otherwise.
        Only rank 0 checks in-flight status and returns result.
        """
        try:
            self._ensure_distributed_init()
            if self.is_tp_rank_0 and self.has_in_flight():
                return False
            self._launch(worker, poc_req, hidden_size=hidden_size)
            return True
        except Exception as e:
            logger.exception("Error launching PoC: %s", e)
            return False

    def _launch(
        self,
        worker: Any,
        poc_req: PoCRequestData,
        *,
        hidden_size: int,
    ) -> None:
        """Fire-and-forget: start PoC forward collectively across all TP ranks.

        Design:
        - All ranks create async thread (required for collective ops).
        - Rank 0 manages event polling and result collection.
        - Each rank uses its own CUDA stream for device-local work.
        """
        from vllm.poc.inference.model_runner import execute_poc_forward

        self._ensure_distributed_init()

        # Force cleanup of any previous thread before starting new one
        # This handles cases where abort() was called but thread is still
        # lingering (gives it final chance to join)
        if self._thread is not None and not self._thread.is_alive():
            self._cleanup_thread()
        elif self._thread is not None and self._thread.is_alive():
            # Thread still running - try aggressive join with shorter timeout
            rank_id = self.tp_group.rank_in_group if self.tp_group else 0
            logger.warning(
                "[Rank %d] Previous PoC thread still running at launch time, "
                "forcing cleanup (may lose results)",
                rank_id,
            )
            self._thread.join(timeout=0.5)
            # Clean up regardless
            self._cleanup_thread()

        # Lazily create per-device CUDA stream
        if self.stream is None:
            self.stream = torch.cuda.Stream(device=worker.device, priority=0)
            logger.info(
                "[Rank %d/%d] Created PoC CUDA stream (device=%s)",
                self.tp_group.rank_in_group if self.tp_group else 0,
                self.tp_group.world_size if self.tp_group else 1,
                worker.device,
            )

        if self.is_tp_rank_0:
            self._pending_request_id = poc_req.request_id
            self._pending_result = None
            self._pending_error = None

        # Clear abort flag for new execution
        self._should_abort.clear()

        assert self.stream is not None, "Stream should be initialized"
        stream = self.stream

        def _run() -> None:
            try:
                # Check abort flag before starting
                if self._should_abort.is_set():
                    logger.info(
                        "[Rank %d] PoC aborted before execution (request %s)",
                        self.tp_group.rank_in_group if self.tp_group else 0,
                        poc_req.request_id,
                    )
                    return

                # Serialize model collectives with normal inference.
                collective_lock = getattr(worker, "_collective_lock", None)
                if collective_lock is None:
                    # Fallback: no lock available (should not happen in v1 GPUWorker).
                    with torch.cuda.stream(stream), torch.inference_mode():
                        result = execute_poc_forward(
                            worker,
                            poc_req.block_hash,
                            poc_req.public_key,
                            poc_req.nonces,
                            poc_req.seq_len,
                            hidden_size,
                            poc_req.k_dim,
                        )
                else:
                    acquired = False
                    # Wait for lock in small increments so abort() can stop quickly.
                    while not acquired and not self._should_abort.is_set():
                        acquired = collective_lock.acquire(timeout=0.1)
                    if not acquired:
                        logger.info(
                            "[Rank %d] PoC aborted while waiting for collective lock (request %s)",
                            self.tp_group.rank_in_group if self.tp_group else 0,
                            poc_req.request_id,
                        )
                        return
                    try:
                        if self._should_abort.is_set():
                            return
                        with torch.cuda.stream(stream), torch.inference_mode():
                            result = execute_poc_forward(
                                worker,
                                poc_req.block_hash,
                                poc_req.public_key,
                                poc_req.nonces,
                                poc_req.seq_len,
                                hidden_size,
                                poc_req.k_dim,
                            )
                    finally:
                        try:
                            collective_lock.release()
                        except Exception:
                            pass

                # Check abort flag after execution
                if self._should_abort.is_set():
                    logger.info(
                        "[Rank %d] PoC aborted after execution (request %s)",
                        self.tp_group.rank_in_group if self.tp_group else 0,
                        poc_req.request_id,
                    )
                    return

                # Record event and store result (rank 0 only)
                if self.is_tp_rank_0:
                    self._event = stream.record_event()
                    self._pending_result = result
            except torch.cuda.OutOfMemoryError as e:
                logger.error(
                    "[Rank %d] PoC forward OOM (request %s): %s",
                    self.tp_group.rank_in_group if self.tp_group else 0,
                    poc_req.request_id,
                    e,
                )
                if self.is_tp_rank_0:
                    self._pending_error = e
            except Exception as e:
                logger.exception(
                    "[Rank %d] PoC forward failed (request %s): %s",
                    self.tp_group.rank_in_group if self.tp_group else 0,
                    poc_req.request_id,
                    e,
                )
                if self.is_tp_rank_0:
                    self._pending_error = e

        # All ranks create threads to avoid deadlock in collective ops
        rank_id = self.tp_group.rank_in_group if self.tp_group else 0
        self._thread = threading.Thread(
            target=_run,
            name=f"poc-forward-rank{rank_id}",
            daemon=True,
        )
        self._thread.start()

    def collect_results(self) -> dict[str, dict] | None:
        """Non-blocking poll: return PoC results if finished; otherwise None.

        Only rank 0 polls events and returns results.
        Non-rank-0 always returns None (scheduler queries rank 0 only).
        """
        try:
            self._ensure_distributed_init()
            if not self.is_tp_rank_0:
                return None
        except Exception as e:
            logger.warning("Error in collect_results initialization: %s", e)
            return None

        if self._thread is None:
            return None

        # Fast path: thread still alive → kernels likely not done.
        if self._thread.is_alive():
            # However, if event is recorded and completed, the thread might only
            # be doing Python-side cleanup. Give it a short chance to finish.
            if self._event is not None and self._event.query():
                self._thread.join(timeout=0.5)
            else:
                return None

        if self._thread.is_alive():
            return None

        request_id = self._pending_request_id
        error = self._pending_error
        result = self._pending_result

        # Reset state.
        self._thread = None
        self._event = None
        self._pending_request_id = None
        self._pending_result = None
        self._pending_error = None

        if request_id is None:
            return {}

        if error is not None:
            logger.warning(
                "PoC request %s completed with error, reporting to scheduler",
                request_id,
            )
            return {request_id: None}  # type: ignore[dict-item]

        if result is None:
            # PP non-last rank or unexpected None.
            return {}

        return {request_id: result}

    def abort(self) -> None:
        """Abort any in-flight PoC execution.

        Sets abort flag and waits for thread to finish.
        Safe to call even if no PoC is running.
        """
        self._should_abort.set()

        if self._thread is not None and self._thread.is_alive():
            rank_id = self.tp_group.rank_in_group if self.tp_group else 0
            logger.info(
                "[Rank %d] Aborting in-flight PoC thread, waiting for cleanup...",
                rank_id,
            )
            # Wait for thread to finish with timeout
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning(
                    "[Rank %d] PoC thread did not finish in 5s, forcing cleanup "
                    "(thread may continue running in background)",
                    rank_id,
                )

        # Force cleanup regardless of thread state
        # This ensures state is clean for next launch
        self._cleanup_thread()
        self._pending_request_id = None
        self._pending_result = None
        self._pending_error = None
