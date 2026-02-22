"""PoC generate queue with bounded nonce cap and result store."""

import asyncio
import contextlib
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from vllm.poc.protocol.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.protocol.enums import CallbackPath, GenerateResultStatus
from vllm.poc.protocol.schemas import (
    GenerateCompletedResponseSchema,
    GeneratedCallbackPayloadSchema,
    GenerateValidatedCompletedResponseSchema,
    ValidatedCallbackPayloadSchema,
)
from vllm.poc.protocol.types import Artifact
from vllm.poc.runtime.callbacks import clear_callback_queue, get_callback_queue
from vllm.poc.runtime.validation_utils import build_encoding, validate_artifacts
from vllm.poc.utils import env
from vllm.poc.utils.poc_logger import init_poc_logger

logger = init_poc_logger(__name__)


@dataclass
class GenerateJob:
    """A queued /generate request."""

    request_id: str
    engine_client: Any
    app_id: int
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: list[int]
    seq_len: int
    k_dim: int
    validation_artifacts: dict[int, str] | None = None
    stat_test_dist_threshold: float = DEFAULT_DIST_THRESHOLD
    stat_test_p_mismatch: float = DEFAULT_P_MISMATCH
    stat_test_fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD
    callback_url: str | None = None
    created_at: float = field(default_factory=time.time)


@dataclass
class GenerateResult:
    """Result record for a queued /generate request."""

    status: GenerateResultStatus
    nonce_count: int = 0
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    result: GenerateCompletedResponseSchema | GenerateValidatedCompletedResponseSchema | None = None
    error: str | None = None


class GenerateQueue:
    """Bounded queue for /generate jobs with result tracking."""

    def __init__(self):
        self._queue: asyncio.Queue[GenerateJob] = asyncio.Queue()
        self._results: dict[str, GenerateResult] = {}
        self._queued_nonces: int = 0
        self._lock: asyncio.Lock = asyncio.Lock()
        self._worker_task: asyncio.Task | None = None
        self._stop_event: asyncio.Event = asyncio.Event()
        self._is_generation_active: Callable[[int], bool] | None = None
        self._callback_queue = None  # Initialized lazily

    def set_generation_active_check(self, fn: Callable[[int], bool]):
        """Set callback to check if /init/generate is active."""
        self._is_generation_active = fn

    @property
    def queued_nonces(self) -> int:
        return self._queued_nonces

    async def enqueue(self, job: GenerateJob) -> str | None:
        """Enqueue a job. Returns None if cap exceeded."""
        async with self._lock:
            new_total = self._queued_nonces + len(job.nonces)
            if new_total > env.POC_MAX_QUEUED_NONCES:
                return None

            self._queued_nonces = new_total
            self._results[job.request_id] = GenerateResult(
                status=GenerateResultStatus.QUEUED,
                nonce_count=len(job.nonces),
            )
            await self._queue.put(job)
            return job.request_id

    def get_result(self, request_id: str) -> GenerateResult | None:
        """Get result for a request_id."""
        return self._results.get(request_id)

    async def clear_all(self):
        """Clear queue and results."""
        async with self._lock:
            while not self._queue.empty():
                try:
                    job = self._queue.get_nowait()
                    if job.request_id in self._results:
                        self._results[job.request_id].status = GenerateResultStatus.CANCELLED
                        self._results[job.request_id].completed_at = time.time()
                except asyncio.QueueEmpty:
                    break

            self._queued_nonces = 0
            self._results.clear()
            self._stop_event.set()

    def cleanup_old_results(self):
        """Remove completed/failed results older than TTL."""
        now = time.time()
        expired = [
            rid
            for rid, rec in self._results.items()
            if rec.status
            in (
                GenerateResultStatus.COMPLETED,
                GenerateResultStatus.FAILED,
                GenerateResultStatus.CANCELLED,
            )
            and rec.completed_at
            and now - rec.completed_at > env.POC_GENERATE_RESULT_TTL_SEC
        ]
        for rid in expired:
            del self._results[rid]

    async def ensure_worker_running(self, engine_client, app_id: int):
        """Ensure the worker task is running."""
        if self._worker_task is None or self._worker_task.done():
            self._stop_event.clear()
            self._worker_task = asyncio.create_task(self._worker_loop(engine_client, app_id))

    async def stop_worker(self):
        """Stop the worker task and callback queue."""
        self._stop_event.set()
        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None

        # Stop callback queue and clear global singleton
        if self._callback_queue:
            await self._callback_queue.stop()
            self._callback_queue = None
        await clear_callback_queue()

    async def _worker_loop(self, engine_client, app_id: int):
        """Background worker that processes queued jobs."""
        # Initialize callback queue with bounded concurrency
        self._callback_queue = get_callback_queue(self._stop_event)
        await self._callback_queue.start()

        logger.info("Generate queue worker started")

        while not self._stop_event.is_set():
            try:
                try:
                    job = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                if job.request_id in self._results:
                    self._results[job.request_id].status = GenerateResultStatus.RUNNING

                try:
                    if self._is_generation_active:
                        while self._is_generation_active(job.app_id):
                            if self._stop_event.is_set():
                                break
                            await asyncio.sleep(0.1)

                    if self._stop_event.is_set():
                        break

                    result = await self._process_job(job)

                    if job.request_id in self._results:
                        self._results[job.request_id].status = GenerateResultStatus.COMPLETED
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].result = result

                    if job.callback_url:
                        # Enqueue callback for delivery with bounded concurrency
                        self._enqueue_callback(job, result)

                except Exception as e:
                    logger.exception(
                        "Generate job %s failed: %s",
                        job.request_id,
                        e,
                    )
                    if job.request_id in self._results:
                        self._results[job.request_id].status = GenerateResultStatus.FAILED
                        self._results[job.request_id].completed_at = time.time()
                        self._results[job.request_id].error = str(e)

                finally:
                    async with self._lock:
                        self._queued_nonces -= len(job.nonces)
                        self._queued_nonces = max(0, self._queued_nonces)

                self.cleanup_old_results()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Generate worker error: %s", e)
                await asyncio.sleep(1)

        logger.info("Generate queue worker stopped")

    async def _process_job(
        self, job: GenerateJob
    ) -> GenerateCompletedResponseSchema | GenerateValidatedCompletedResponseSchema:
        """Process a single generate job."""
        total_nonces = len(job.nonces)
        logger.info("PoC queue job %s: %d nonces", job.request_id[:8], total_nonces)

        start_time = time.time()
        computed_artifacts: list[Artifact] = []

        while True:
            if self._stop_event.is_set():
                raise RuntimeError("Job cancelled")

            if self._is_generation_active and self._is_generation_active(job.app_id):
                await asyncio.sleep(0.1)
                continue

            try:
                from .routes import run_poc_request

                artifacts = await asyncio.wait_for(
                    run_poc_request(
                        job.engine_client,
                        job.nonces,
                        job.block_hash,
                        job.public_key,
                        job.seq_len,
                        job.k_dim,
                    ),
                    timeout=env.POC_GENERATE_CHUNK_TIMEOUT_SEC,
                )
            except asyncio.CancelledError:
                logger.info(
                    "PoC queue job %s: cancelled during RPC",
                    job.request_id[:8],
                )
                raise RuntimeError("Job cancelled") from None
            except asyncio.TimeoutError as e:
                raise RuntimeError("Timeout waiting for engine RPC") from e

            computed_artifacts.extend(artifacts)
            break

        elapsed = time.time() - start_time
        rate = total_nonces / elapsed if elapsed > 0 else 0
        logger.info(
            "PoC queue job %s completed: %d nonces in %.2fs (%.0f/s)",
            job.request_id[:8],
            total_nonces,
            elapsed,
            rate,
        )

        if job.validation_artifacts is None:
            return GenerateCompletedResponseSchema(
                request_id=job.request_id,
                artifacts=computed_artifacts,
                encoding=build_encoding(job.k_dim),
            )
        validation = validate_artifacts(
            computed_artifacts,
            job.validation_artifacts,
            dist_threshold=job.stat_test_dist_threshold,
            p_mismatch=job.stat_test_p_mismatch,
            fraud_threshold=job.stat_test_fraud_threshold,
            k_dim=job.k_dim,
        )
        return GenerateValidatedCompletedResponseSchema(
            request_id=job.request_id,
            n_total=validation.n_total,
            n_mismatch=validation.n_mismatch,
            mismatch_nonces=validation.mismatch_nonces,
            p_value=validation.p_value,
            fraud_detected=validation.fraud_detected,
        )

    def _enqueue_callback(
        self,
        job: GenerateJob,
        result: (GenerateCompletedResponseSchema | GenerateValidatedCompletedResponseSchema),
    ):
        """Enqueue callback for delivery via bounded callback queue."""
        if self._callback_queue is None:
            logger.warning(
                "Callback queue not initialized, skipping callback for %s",
                job.request_id,
            )
            return

        if isinstance(result, GenerateCompletedResponseSchema):
            payload = GeneratedCallbackPayloadSchema(
                request_id=job.request_id,
                block_hash=job.block_hash,
                block_height=job.block_height,
                public_key=job.public_key,
                node_id=job.node_id,
                artifacts=result.artifacts,
                encoding=result.encoding,
            )
            self._callback_queue.enqueue(
                job.callback_url,
                CallbackPath.GENERATED,
                payload,
            )
            return

        payload = ValidatedCallbackPayloadSchema(
            request_id=job.request_id,
            block_hash=job.block_hash,
            block_height=job.block_height,
            public_key=job.public_key,
            node_id=job.node_id,
            n_total=result.n_total,
            n_mismatch=result.n_mismatch,
            mismatch_nonces=result.mismatch_nonces,
            p_value=result.p_value,
            fraud_detected=result.fraud_detected,
        )
        self._callback_queue.enqueue(job.callback_url, CallbackPath.VALIDATED, payload)


_queue_instance: GenerateQueue | None = None


def get_queue() -> GenerateQueue:
    """Get or create singleton queue instance."""
    global _queue_instance
    if _queue_instance is None:
        _queue_instance = GenerateQueue()
    return _queue_instance


async def clear_queue():
    """Clear the queue singleton."""
    global _queue_instance
    if _queue_instance:
        await _queue_instance.clear_all()
        await _queue_instance.stop_worker()
        _queue_instance = None
