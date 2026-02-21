# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""
PoCScheduler: Isolated scheduler for Proof-of-Compute requests.

Manages PoC request queue, execution state, and result collection independently
from the main inference scheduler to avoid resource conflicts and ensure proper
thread-safety with AsyncPoCWorker.
"""

import time
from collections import defaultdict
from typing import TYPE_CHECKING, Callable, Optional

from vllm.poc.utils.poc_logger import init_poc_logger
from vllm.poc.v1.request import PoCRequest
from vllm.v1.core.sched.output import PoCRequestData
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_poc_logger(__name__)


class PoCScheduler:
    """
    Dedicated scheduler for Proof-of-Compute (PoC) requests.

    Manages:
    - PoC waiting queue (separate from main inference queue)
    - Currently running PoC request state
    - Async result collection with timeout handling
    - Request lifecycle (add, schedule, abort, finish)

    Design principles:
    - Isolated from main scheduler to prevent KV cache conflicts
    - Thread-safe coordination with AsyncPoCWorker via status tracking
    - Timeout protection (30s) to prevent perpetual blocking
    """

    def __init__(
        self,
        policy: SchedulingPolicy,
        kv_cache_config: "KVCacheConfig",
    ) -> None:
        """
        Initialize PoC scheduler.

        Args:
            policy: Scheduling policy (e.g., FCFS, PRIORITY) for PoC queue
            kv_cache_config: KV cache configuration (for metadata in outputs)
        """
        self.policy = policy
        self.kv_cache_config = kv_cache_config
        self._abort_poc_fn: Optional[Callable[[], None]] = None

        # PoC waiting queue: uses same priority semantics as main scheduler
        # (smaller priority => higher priority)
        self.poc_waiting = create_request_queue(self.policy)

        # All PoC requests by ID (waiting + running)
        self.poc_requests: dict[str, PoCRequest] = {}

        # Currently executing PoC request (background thread via AsyncPoCWorker)
        self._poc_running: PoCRequest | None = None
        self._poc_running_start_time: float | None = None

        # Timeout for stuck PoC executions (seconds)
        self._poc_timeout: float = 30.0

    def set_abort_poc_fn(self, abort_fn: Callable[[], None]) -> None:
        """Set the RPC function to abort PoC on worker."""
        self._abort_poc_fn = abort_fn

    def has_pending_requests(self) -> bool:
        """Check if any PoC request is waiting to be scheduled."""
        return bool(self.poc_waiting)

    def can_schedule(self) -> bool:
        """Check if we can schedule a new PoC request (no PoC currently running)."""
        return self.has_pending_requests() and self._poc_running is None

    def schedule_next(self) -> PoCRequestData | None:
        """
        Schedule the next PoC request from waiting queue.

        Returns:
            PoCRequestData if a request was scheduled, None otherwise
        """
        if not self.can_schedule():
            return None

        poc_req = self.poc_waiting.pop_request()
        self._poc_running = poc_req
        self._poc_running_start_time = time.monotonic()
        poc_req.status = RequestStatus.RUNNING

        return PoCRequestData(
            request_id=poc_req.request_id,
            block_hash=poc_req.poc_params.block_hash,
            public_key=poc_req.poc_params.public_key,
            nonces=poc_req.poc_params.nonces,
            seq_len=poc_req.poc_params.seq_len,
            k_dim=poc_req.poc_params.k_dim,
            priority=poc_req.priority,
        )

    def collect_async_results(
        self,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, list[EngineCoreOutput]] | None:
        """
        Collect PoC results from ModelRunnerOutput (async from AsyncPoCWorker).

        Args:
            model_runner_output: Output from worker containing poc_results dict

        Returns:
            Dict mapping client_index -> [EngineCoreOutput] if PoC completed,
            None if still running or no PoC active
        """
        poc_req = self._poc_running
        if poc_req is None:
            return None

        poc_result = None
        if model_runner_output.poc_results is not None:
            poc_result = model_runner_output.poc_results.get(poc_req.request_id)

        # Results dict exists but doesn't have our request — might be
        # an error marker (request_id → None)
        if (
            poc_result is None
            and model_runner_output.poc_results
            and poc_req.request_id in model_runner_output.poc_results
        ):
            poc_result = model_runner_output.poc_results[poc_req.request_id]

        if poc_result is None and not model_runner_output.poc_results:
            # No results at all in this iteration — PoC still running
            # Check timeout to prevent perpetual blocking
            if self._poc_running_start_time is not None:
                elapsed = time.monotonic() - self._poc_running_start_time
                if elapsed > self._poc_timeout:
                    logger.warning(
                        "PoC request %s timed out after %.1fs; clearing stale state",
                        poc_req.request_id,
                        elapsed,
                    )
                    # Clear stuck PoC and return error
                    self._poc_running = None
                    self._poc_running_start_time = None
                    poc_req.status = RequestStatus.FINISHED_ERROR
                    self.poc_requests.pop(poc_req.request_id, None)

                    outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)
                    outputs[poc_req.client_index].append(
                        EngineCoreOutput(
                            request_id=poc_req.request_id,
                            new_token_ids=[],
                            finish_reason=FinishReason.ERROR,
                            poc_result=None,
                        )
                    )
                    return dict(outputs)
            return None

        # PoC completed (success or error)
        self._poc_running = None
        self._poc_running_start_time = None

        outputs: dict[int, list[EngineCoreOutput]] = defaultdict(list)

        if poc_result is None:
            # Error: worker returned None result for this request
            poc_req.status = RequestStatus.FINISHED_ERROR
            self.poc_requests.pop(poc_req.request_id, None)
            outputs[poc_req.client_index].append(
                EngineCoreOutput(
                    request_id=poc_req.request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.ERROR,
                    poc_result=None,
                )
            )
        else:
            # Success
            poc_req.status = RequestStatus.FINISHED_STOPPED
            self.poc_requests.pop(poc_req.request_id, None)
            outputs[poc_req.client_index].append(
                EngineCoreOutput(
                    request_id=poc_req.request_id,
                    new_token_ids=[],
                    finish_reason=FinishReason.STOP,
                    poc_result=poc_result,
                )
            )

        return dict(outputs)

    def add_request(self, request: PoCRequest) -> None:
        """
        Add a new PoC request to the waiting queue.

        Args:
            request: PoC request to add

        Raises:
            ValueError: If request ID already exists
        """
        if request.request_id in self.poc_requests:
            raise ValueError(f"duplicate PoC request id: {request.request_id}")

        self.poc_waiting.add_request(request)  # type: ignore[arg-type]
        self.poc_requests[request.request_id] = request

    def abort_requests(
        self,
        request_ids: set[str],
        finished_status: RequestStatus,
    ) -> None:
        """
        Abort PoC requests by ID.

        Args:
            request_ids: Set of request IDs to abort
            finished_status: Status to mark finished requests (e.g., FINISHED_ABORTED)
        """
        assert RequestStatus.is_finished(finished_status)

        poc_to_remove_waiting: list[PoCRequest] = []
        
        if request_ids:
            logger.info("PoC abort_requests called for %d request(s)", len(request_ids))

        for req_id in request_ids:
            poc_req = self.poc_requests.get(req_id)
            if poc_req is None or poc_req.is_finished():
                continue

            if (
                poc_req.status == RequestStatus.RUNNING
                and self._poc_running is not None
            ):
                if self._poc_running.request_id == req_id:
                    # Abort currently running PoC
                    logger.info(
                        "Aborting running PoC request %s, calling worker abort via RPC",
                        req_id,
                    )
                    self._poc_running.status = finished_status
                    self._poc_running = None
                    self._poc_running_start_time = None
                    # Signal worker to abort in-flight execution via RPC
                    if self._abort_poc_fn is not None:
                        try:
                            logger.info("Calling abort_poc_fn() on worker")
                            self._abort_poc_fn()
                            logger.info("Worker abort completed")
                        except Exception as e:
                            logger.warning("Failed to abort PoC on worker: %s", e)
                    else:
                        logger.warning(
                            "PoC abort_poc_fn is None - abort coordination not set up"
                        )
            else:
                # Abort waiting PoC
                logger.info("Aborting waiting PoC request %s", req_id)
                poc_req.status = finished_status
                poc_to_remove_waiting.append(poc_req)

        # Remove from waiting queue and registry
        if poc_to_remove_waiting:
            self.poc_waiting.remove_requests(poc_to_remove_waiting)
        for req in poc_to_remove_waiting:
            self.poc_requests.pop(req.request_id, None)

    def get_num_unfinished_requests(self) -> tuple[int, int]:
        """
        Get number of running and waiting PoC requests.

        Returns:
            (num_running, num_waiting) tuple
        """
        poc_running = 1 if self._poc_running is not None else 0
        return poc_running, len(self.poc_waiting)
