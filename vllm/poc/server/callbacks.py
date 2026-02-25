"""PoC callback sender with retry-until-stop and bounded buffer."""

import asyncio
import contextlib
import json
import time
from collections import deque
from typing import Any

import aiohttp
from pydantic import BaseModel

import vllm.poc.env as env
from vllm.poc._log import init_poc_logger
from vllm.poc.constants import (
    DEFAULT_K_DIM,
    POC_CALLBACK_RETRY_BACKOFF_SEC,
    POC_CALLBACK_RETRY_MAX_BACKOFF_SEC,
)
from vllm.poc.server.models import Artifact, ArtifactBatchMeta, CallbackPath
from vllm.poc.server.schemas import ArtifactBatchSchema
from vllm.poc.server.validation import build_encoding

logger = init_poc_logger(__name__)


def _maybe_log_artifacts_json(payload: dict[str, Any], sink: str) -> None:
    """Optionally log full callback payload JSON for generated artifacts.

    Controlled by env vars:
    - POC_LOG_ARTIFACTS_JSON=1: log full JSON payload via logger.info
    """

    if "artifacts" not in payload:
        return
    if not env.POC_LOG_ARTIFACTS_JSON:
        return

    try:
        payload_json = json.dumps(payload, ensure_ascii=False)
        logger.info("Artifacts payload (%s): %s", sink, payload_json)
    except Exception as e:
        logger.warning("Failed to log artifacts JSON (%s): %s", sink, e)


FALLBACK_BLOCK_HASH = ""
FALLBACK_BLOCK_HEIGHT = 0
FALLBACK_PUBLIC_KEY = ""
FALLBACK_NODE_ID = 0


class CallbackSender:
    """Manages callback sending with retry and bounded buffer."""

    def __init__(
        self,
        callback_url: str,
        stop_event: asyncio.Event,
        k_dim: int = DEFAULT_K_DIM,
        max_artifacts: int | None = None,
    ):
        self.callback_url = callback_url
        self.stop_event = stop_event
        self.k_dim = k_dim
        self.max_artifacts = max_artifacts or env.POC_CALLBACK_MAX_ARTIFACTS

        self._buffer: deque[Artifact] = deque()
        self._metadata: ArtifactBatchMeta | None = None
        self._pending_payload: ArtifactBatchSchema | None = None
        self._task: asyncio.Task | None = None

    def add_artifacts(self, artifacts: list[Artifact], metadata: ArtifactBatchMeta):
        """Add artifacts to buffer, dropping oldest if cap exceeded."""

        self._metadata = metadata
        for artifact in artifacts:
            self._buffer.append(artifact)

        while len(self._buffer) > self.max_artifacts:
            self._buffer.popleft()

    def clear(self):
        """Clear all buffered artifacts."""

        self._buffer.clear()
        self._pending_payload = None

    @property
    def buffered_count(self) -> int:
        return len(self._buffer)

    async def run(self):
        """Main sender loop - batches and sends with retry-until-stop."""

        last_send_time = time.time()
        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
        retry_attempt = 0

        async with aiohttp.ClientSession() as session:
            while not self.stop_event.is_set():
                await asyncio.sleep(0.1)

                current_time = time.time()
                should_send = (self._buffer or self._pending_payload) and (
                    current_time - last_send_time >= env.POC_CALLBACK_INTERVAL_SEC
                )

                if not should_send:
                    continue

                if self._pending_payload is None and self._buffer:
                    artifacts_to_send = list(self._buffer)
                    self._buffer.clear()
                    self._pending_payload = ArtifactBatchSchema(
                        public_key=(
                            self._metadata.public_key
                            if self._metadata is not None
                            else FALLBACK_PUBLIC_KEY
                        ),
                        block_hash=(
                            self._metadata.block_hash
                            if self._metadata is not None
                            else FALLBACK_BLOCK_HASH
                        ),
                        block_height=(
                            self._metadata.block_height
                            if self._metadata is not None
                            else FALLBACK_BLOCK_HEIGHT
                        ),
                        node_id=(
                            self._metadata.node_id
                            if self._metadata is not None
                            else FALLBACK_NODE_ID
                        ),
                        artifacts=artifacts_to_send,
                        encoding=build_encoding(self.k_dim),
                    )
                    retry_attempt = 0

                if self._pending_payload:
                    retry_attempt += 1
                    payload_dict = self._pending_payload.model_dump(mode="json")
                    success = await self._send_callback(
                        session, payload_dict, retry_attempt
                    )
                    if success:
                        if retry_attempt > 1:
                            logger.info(
                                "Callback to %s succeeded after %d attempts",
                                self.callback_url,
                                retry_attempt,
                            )
                        self._pending_payload = None
                        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
                        retry_attempt = 0
                        last_send_time = current_time
                    elif retry_attempt >= env.POC_CALLBACK_MAX_RETRIES:
                        n_artifacts = len(payload_dict.get("artifacts", []))
                        logger.error(
                            "Callback to %s failed after %d attempts, "
                            "dropping %d artifacts",
                            self.callback_url,
                            retry_attempt,
                            n_artifacts,
                        )
                        self._pending_payload = None
                        backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
                        retry_attempt = 0
                        last_send_time = current_time
                    else:
                        logger.warning(
                            "Callback to %s failed (attempt %d/%d, backoff %.1fs)",
                            self.callback_url,
                            retry_attempt,
                            env.POC_CALLBACK_MAX_RETRIES,
                            backoff,
                        )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)

    async def _send_callback(
        self, session: aiohttp.ClientSession, payload: dict, attempt: int = 1
    ) -> bool:
        """Send callback, return True on success."""

        _ = attempt
        _maybe_log_artifacts_json(payload, "callback_sender")
        try:
            async with session.post(
                f"{self.callback_url}/generated",
                json=payload,
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status < 400:
                    logger.debug(
                        "Callback sent: %d artifacts",
                        len(payload.get("artifacts", [])),
                    )
                    return True
                return False
        except Exception:
            return False


class CallbackQueue:
    """Queue for reliable callback delivery with bounded concurrency."""

    def __init__(
        self,
        stop_event: asyncio.Event,
        max_concurrent: int | None = None,
        max_queue_size: int | None = None,
    ):
        self.stop_event = stop_event
        self.max_concurrent = max_concurrent or env.POC_CALLBACK_MAX_CONCURRENT
        self.max_queue_size = max_queue_size or env.POC_CALLBACK_QUEUE_SIZE

        self._queue: deque[tuple[str, CallbackPath, BaseModel]] = deque(
            maxlen=self.max_queue_size
        )
        self._semaphore = asyncio.Semaphore(self.max_concurrent)
        self._active_tasks: set[asyncio.Task] = set()
        self._worker_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._dropped_count = 0

    def enqueue(self, url: str, path: CallbackPath, payload: BaseModel):
        """Add callback to queue. Drops oldest if queue is full."""

        was_full = len(self._queue) >= self.max_queue_size
        self._queue.append((url, path, payload))
        if was_full:
            self._dropped_count += 1
            if self._dropped_count == 1 or self._dropped_count % 100 == 0:
                logger.warning(
                    "Callback queue full, dropped %d callbacks total",
                    self._dropped_count,
                )

    @property
    def pending_count(self) -> int:
        return len(self._queue)

    @property
    def active_count(self) -> int:
        return len(self._active_tasks)

    async def start(self):
        """Start the callback worker."""

        if self._worker_task is None or self._worker_task.done():
            self._session = aiohttp.ClientSession()
            self._worker_task = asyncio.create_task(self._worker_loop())
            logger.info(
                "Callback queue started (max_concurrent=%d, max_queue=%d)",
                self.max_concurrent,
                self.max_queue_size,
            )

    async def stop(self):
        """Stop the callback worker and cleanup."""

        if self._worker_task and not self._worker_task.done():
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task

        for task in list(self._active_tasks):
            task.cancel()
        if self._active_tasks:
            await asyncio.gather(*self._active_tasks, return_exceptions=True)
        self._active_tasks.clear()

        if self._session:
            await self._session.close()
            self._session = None

        remaining = len(self._queue)
        if remaining > 0:
            logger.warning(
                "Callback queue stopped with %d pending callbacks",
                remaining,
            )
        self._queue.clear()

    async def _worker_loop(self):
        """Main worker loop - dispatches callbacks with bounded concurrency."""

        logger.info("Callback worker loop starting")
        try:
            while not self.stop_event.is_set():
                self._active_tasks = {t for t in self._active_tasks if not t.done()}

                if not self._queue:
                    await asyncio.sleep(0.05)
                    continue

                url, path, payload = self._queue.popleft()
                task = asyncio.create_task(self._send_with_retry(url, path, payload))
                self._active_tasks.add(task)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.exception("Callback worker loop crashed: %s", e)
        logger.info("Callback worker loop exited")

    async def _send_with_retry(
        self, url: str, path: CallbackPath, payload: BaseModel
    ) -> bool:
        """Send callback with exponential backoff retry."""

        assert self._session is not None
        async with self._semaphore:
            payload_dict = payload.model_dump(mode="json")
            _maybe_log_artifacts_json(payload_dict, f"callback_queue:{path.value}")
            backoff = POC_CALLBACK_RETRY_BACKOFF_SEC
            attempt = 0
            url_path = f"{url}/{path.value}"

            while attempt < env.POC_CALLBACK_MAX_RETRIES:
                attempt += 1
                if self.stop_event.is_set():
                    return False

                try:
                    async with self._session.post(
                        url_path,
                        json=payload_dict,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status < 400:
                            if attempt > 1:
                                logger.info(
                                    "Callback to %s succeeded after %d attempts",
                                    url_path,
                                    attempt,
                                )
                            return True
                        logger.warning(
                            "Callback to %s HTTP %d (attempt %d/%d)",
                            url_path,
                            resp.status,
                            attempt,
                            env.POC_CALLBACK_MAX_RETRIES,
                        )
                except Exception as e:
                    logger.warning(
                        "Callback to %s failed: %s (attempt %d/%d)",
                        url_path,
                        e,
                        attempt,
                        env.POC_CALLBACK_MAX_RETRIES,
                    )

                if attempt < env.POC_CALLBACK_MAX_RETRIES:
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, POC_CALLBACK_RETRY_MAX_BACKOFF_SEC)

            logger.error(
                "Callback to %s failed after %d attempts, giving up",
                url_path,
                env.POC_CALLBACK_MAX_RETRIES,
            )
            return False


_callback_queue: CallbackQueue | None = None


def get_callback_queue(stop_event: asyncio.Event) -> CallbackQueue:
    """Get or create singleton callback queue."""

    global _callback_queue
    if _callback_queue is None:
        _callback_queue = CallbackQueue(stop_event)
    return _callback_queue


async def clear_callback_queue():
    """Stop and clear the callback queue singleton."""

    global _callback_queue
    if _callback_queue:
        await _callback_queue.stop()
        _callback_queue = None
