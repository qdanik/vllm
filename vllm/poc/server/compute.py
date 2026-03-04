"""PoC nonce computation and generation loop.

Combines the former ``api/compute`` and ``api/generation`` modules into
one coherent module that handles both the stateless compute path and the
long-running ``/init/generate`` loop.
"""

import asyncio
import time
import uuid
from typing import Any

import vllm.poc.env as env
from vllm.poc._log import init_poc_logger
from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC, POC_REQUEST_PRIORITY
from vllm.poc.server.callbacks import CallbackSender
from vllm.poc.server.models import (
    Artifact,
    ArtifactBatchMeta,
    NonceIterator,
    PoCConfig,
    PoCGenerationStats,
)

logger = init_poc_logger(__name__)


def _resolve_generation_batch_size(config: PoCConfig) -> int:
    default_batch_size = env.POC_BATCH_SIZE_DEFAULT
    requested_batch_size = config.batch_size

    if env.POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT:
        if (
            requested_batch_size is not None
            and requested_batch_size != default_batch_size
        ):
            logger.info(
                "Ignoring init batch_size=%s due to force flag. Currently set to %s",
                requested_batch_size,
                default_batch_size,
            )
        return default_batch_size

    if requested_batch_size is None:
        return default_batch_size

    if requested_batch_size <= 0:
        logger.warning(
            "Invalid init batch_size=%s; falling back to %s",
            requested_batch_size,
            default_batch_size,
        )
        return default_batch_size

    return requested_batch_size


def _generate_request_id() -> str:
    return str(uuid.uuid4())


async def compute_artifact(
    engine_client,
    nonces: list[int],
    block_hash: str,
    block_height: int,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float | None = None,
) -> list[Artifact]:
    """Run PoC forward via first-class scheduler request and return artifacts."""

    if timeout_sec is None:
        timeout_sec = env.POC_GENERATE_CHUNK_TIMEOUT_SEC

    if not nonces:
        return []

    async def _run_one(nonce: int) -> dict[str, Any]:
        request_id = _generate_request_id()
        return await engine_client.poc_compute(
            request_id=request_id,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout=None,
            priority=POC_REQUEST_PRIORITY,
        )

    # Fire all nonces concurrently — the scheduler auto-batches them into
    # steps based on its own token budget, so client-side chunking/semaphore
    # only adds latency without improving throughput.
    async def _run_all() -> list[dict[str, Any]]:
        return list(await asyncio.gather(*(_run_one(n) for n in nonces)))

    poc_task = asyncio.create_task(_run_all())

    try:
        if timeout_sec is None:
            results: list[dict[str, Any]] = await poc_task
        else:
            local_timeout_count = 0
            while True:
                try:
                    results = await asyncio.wait_for(
                        asyncio.shield(poc_task), timeout=timeout_sec
                    )
                    break
                except asyncio.TimeoutError:
                    if poc_task.done():
                        results = await poc_task
                        break
                    local_timeout_count += 1
                    if local_timeout_count == 1 or local_timeout_count % 10 == 0:
                        logger.warning(
                            "PoC still pending after %.1fs (#%d); "
                            "engine likely busy",
                            timeout_sec,
                            local_timeout_count,
                        )
                    await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                    continue
    except asyncio.CancelledError:
        poc_task.cancel()
        raise
    except Exception:
        logger.exception(
            "PoC request failed (block_hash=%s, nonces=%s)",
            block_hash,
            nonces,
        )
        raise

    all_nonces: list[int] = []
    all_vectors_b64: list[str] = []
    for result in results:
        if not result:
            continue
        all_nonces.extend(result.get("nonces", []))
        all_vectors_b64.extend(result.get("vectors_b64", []))

    return [
        Artifact(nonce=int(nonce), vector_b64=str(vector_b64))
        for nonce, vector_b64 in zip(all_nonces, all_vectors_b64)
    ]


async def compute_artifacts_chunk(
    engine_client,
    nonces: list[int],
    block_hash: str,
    block_height: int,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float | None = None,
) -> list[Artifact]:
    try:
        return await compute_artifact(
            engine_client=engine_client,
            nonces=nonces,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout_sec=timeout_sec,
        )
    except TimeoutError as e:
        raise RuntimeError(f"Timeout after {timeout_sec}s") from e


async def generation_loop(
    engine_client,
    stop_event: asyncio.Event,
    callback_sender: CallbackSender | None,
    config: PoCConfig,
    stats: PoCGenerationStats,
):
    nonce_iter = NonceIterator(
        node_id=config.node_id,
        n_nodes=config.node_count,
        group_id=config.group_id,
        n_groups=config.n_groups,
    )

    start_time = time.time()
    stats.start_time = start_time
    stats.total_processed = 0
    last_report_time = start_time
    last_report_total = 0

    logger.info(
        "Generation started (node %s/%s, group %s/%s)",
        config.node_id,
        config.node_count,
        config.group_id,
        config.n_groups,
    )
    timeout_count = 0
    error_count = 0
    pending_nonces: list[int] | None = None
    batch_size = _resolve_generation_batch_size(config)

    logger.info("Generation loop batch_size=%s", batch_size)

    try:
        while not stop_event.is_set():
            nonces = (
                pending_nonces
                if pending_nonces is not None
                else nonce_iter.take(batch_size)
            )

            try:
                artifacts = await compute_artifact(
                    engine_client=engine_client,
                    nonces=nonces,
                    block_hash=config.block_hash,
                    block_height=config.block_height,
                    public_key=config.public_key,
                    seq_len=config.seq_len,
                    k_dim=config.k_dim,
                    timeout_sec=env.POC_RPC_TIMEOUT_MS / 1000.0,
                )
                timeout_count = 0
                error_count = 0
            except TimeoutError:
                timeout_count += 1
                if timeout_count == 1 or timeout_count % 10 == 0:
                    logger.warning(
                        "Generation timed out (#%d), engine busy",
                        timeout_count,
                    )
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue
            except Exception:
                error_count += 1
                if error_count == 1 or error_count % 10 == 0:
                    logger.warning(
                        "Generation request failed (#%d), backing off",
                        error_count,
                    )
                pending_nonces = nonces
                await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                continue

            pending_nonces = None

            if artifacts and callback_sender:
                callback_sender.add_artifacts(
                    artifacts,
                    ArtifactBatchMeta(
                        public_key=config.public_key,
                        block_hash=config.block_hash,
                        block_height=config.block_height,
                        node_id=config.node_id,
                    ),
                )

            stats.total_processed += len(artifacts)

            current_time = time.time()
            if current_time - last_report_time >= 5.0:
                window_sec = current_time - last_report_time
                window_delta = stats.total_processed - last_report_total
                rate = (window_delta / (window_sec / 60.0)) if window_sec > 0 else 0
                logger.info(
                    "Generated: %d nonces (%.0f/min)",
                    stats.total_processed,
                    rate,
                )
                last_report_time = current_time
                last_report_total = stats.total_processed

    except asyncio.CancelledError:
        elapsed_min = (time.time() - start_time) / 60
        logger.info(
            "Generation stopped: %d nonces in %.2fmin",
            stats.total_processed,
            elapsed_min,
        )
    except Exception as e:
        logger.exception("Generation crashed: %s", e)
        raise
