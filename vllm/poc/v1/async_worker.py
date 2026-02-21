# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Async PoC worker state (v1).

This module encapsulates the fire-and-forget PoC execution state used by the
v1 GPU worker:

- PoC runs in a background Python thread.
- GPU work is enqueued onto a dedicated CUDA stream.
- Completion is detected via a CUDA event that can be polled with `event.query()`.

The intent is to keep the main `Worker` class readable while ensuring:
- normal inference never blocks on PoC
- PoC failures (incl. CUDA OOM) are isolated and reported as request-level errors
"""

from __future__ import annotations

import threading
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.v1.core.sched.output import PoCRequestData

logger = init_logger(__name__)


class AsyncPoCWorker:
    """Holds all mutable state for async PoC execution on a worker."""

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self._event: torch.cuda.Event | None = None
        self._thread: threading.Thread | None = None

        self._pending_result: dict[str, Any] | None = None
        self._pending_error: BaseException | None = None
        self._pending_request_id: str | None = None

    def has_in_flight(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def launch_if_idle(
        self,
        worker: Any,
        poc_req: PoCRequestData,
        *,
        hidden_size: int,
    ) -> bool:
        """Launch PoC if no previous PoC is currently in-flight.

        Returns True if a new PoC was launched, False otherwise.
        """
        if self.has_in_flight():
            return False
        self._launch(worker, poc_req, hidden_size=hidden_size)
        return True

    def _launch(
        self,
        worker: Any,
        poc_req: PoCRequestData,
        *,
        hidden_size: int,
    ) -> None:
        """Fire-and-forget: start PoC forward in a background thread + CUDA stream."""
        from vllm.poc.inference.model_runner import execute_poc_forward

        # Lazily create a dedicated CUDA stream and reuse it across PoC runs.
        if self._stream is None:
            # `worker.device` is set after `init_device()`.
            self._stream = torch.cuda.Stream(device=worker.device)

        self._pending_request_id = poc_req.request_id
        self._pending_result = None
        self._pending_error = None

        stream = self._stream

        def _run() -> None:
            try:
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

                # Record an event so the main thread can poll for completion
                # without calling torch.cuda.synchronize().
                self._event = stream.record_event()
                self._pending_result = result
            except torch.cuda.OutOfMemoryError as e:
                logger.error(
                    "PoC forward OOM (request %s): %s",
                    poc_req.request_id,
                    e,
                )
                self._pending_error = e
            except Exception as e:
                logger.exception(
                    "PoC forward failed (request %s): %s",
                    poc_req.request_id,
                    e,
                )
                self._pending_error = e

        self._thread = threading.Thread(target=_run, name="poc-forward", daemon=True)
        self._thread.start()

    def collect_results(self) -> dict[str, dict] | None:
        """Non-blocking poll: return PoC results if finished; otherwise None.

        If the PoC thread failed, returns a dict keyed by request_id with value
        None so the scheduler can route it as an ERROR.
        """
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
