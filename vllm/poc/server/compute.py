"""PoC nonce computation and generation loop.

Combines the former ``api/compute`` and ``api/generation`` modules into
one coherent module that handles both the stateless compute path and the
long-running ``/init/generate`` loop.
"""

import asyncio
import base64
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
    RawArtifact,
)

logger = init_poc_logger(__name__)

type PoCRawResult = dict[str, Any]


def _resolve_generation_batch_size(engine_client, config: PoCConfig) -> int:
    default = env.POC_BATCH_SIZE_DEFAULT
    requested = config.batch_size

    if env.POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT:
        if requested is not None and requested != default:
            logger.info(
                "Ignoring init batch_size=%s due to force flag. Currently set to %s",
                requested,
                default,
            )
        batch = default
    elif requested and requested > 0:
        batch = requested
    else:
        if requested is not None:
            logger.warning(
                "Invalid init batch_size=%s; falling back to %s", requested, default
            )
        batch = default

    return batch


def _generate_request_id() -> str:
    return str(uuid.uuid4())


def _extract_vectors_b64(result: PoCRawResult) -> list[str]:
    vectors_b64 = result.get("vectors_b64")
    if vectors_b64 is not None:
        return [str(vector) for vector in vectors_b64]

    vectors_bin = result.get("vectors_bin")
    if vectors_bin is None:
        return []

    encoded: list[str] = []
    for raw in vectors_bin:
        if isinstance(raw, memoryview):
            raw_bytes = raw.tobytes()
        elif isinstance(raw, bytearray):
            raw_bytes = bytes(raw)
        elif isinstance(raw, bytes):
            raw_bytes = raw
        else:
            raise TypeError(f"Unsupported vectors_bin item type: {type(raw)!r}")
        encoded.append(base64.b64encode(raw_bytes).decode("ascii"))
    return encoded


def _build_artifacts(results: list[PoCRawResult]) -> list[Artifact]:
    all_nonces: list[int] = []
    all_vectors_b64: list[str] = []
    for result in results:
        if not result:
            continue
        all_nonces.extend(result.get("nonces", []))
        all_vectors_b64.extend(_extract_vectors_b64(result))

    return [
        Artifact(nonce=int(nonce), vector_b64=str(vector_b64))
        for nonce, vector_b64 in zip(all_nonces, all_vectors_b64)
    ]


def _build_callback_artifacts(results: list[PoCRawResult]) -> list[RawArtifact]:
    callback_artifacts: list[RawArtifact] = []
    for result in results:
        if not result:
            continue

        nonces = result.get("nonces", [])
        vectors_bin = result.get("vectors_bin")
        if vectors_bin is not None:
            for nonce, raw in zip(nonces, vectors_bin):
                if isinstance(raw, memoryview):
                    raw_bytes = raw.tobytes()
                elif isinstance(raw, bytearray):
                    raw_bytes = bytes(raw)
                elif isinstance(raw, bytes):
                    raw_bytes = raw
                else:
                    raise TypeError(
                        f"Unsupported vectors_bin item type: {type(raw)!r}"
                    )
                callback_artifacts.append(
                    RawArtifact(nonce=int(nonce), vector_bin=raw_bytes)
                )
            continue

        for nonce, vector_b64 in zip(nonces, _extract_vectors_b64(result)):
            callback_artifacts.append(
                RawArtifact(nonce=int(nonce), vector_b64=str(vector_b64))
            )

    return callback_artifacts


def _count_result_nonces(results: list[PoCRawResult]) -> int:
    total = 0
    for result in results:
        if result:
            total += len(result.get("nonces", []))
    return total


async def compute_artifact(
    engine_client,
    nonces: list[int],
    block_hash: str,
    block_height: int,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float | None = None,
) -> list[PoCRawResult]:
    """Run PoC forward via batch scheduler request and return raw results.

    Uses ``poc_compute_batch`` to pre-build all requests and submit
    them in a tight loop, avoiding 128 separate asyncio coroutine
    creations and interleaved yields.
    """

    if timeout_sec is None:
        timeout_sec = env.POC_GENERATE_CHUNK_TIMEOUT_SEC

    if not nonces:
        return []

    request_ids = [_generate_request_id() for _ in nonces]

    poc_batch_fn = None
    if hasattr(type(engine_client), "poc_compute_batch"):
        poc_batch_fn = getattr(engine_client, "poc_compute_batch")

    if poc_batch_fn is not None:
        # Fast path: single batch submission.
        poc_task = asyncio.create_task(poc_batch_fn(
            request_ids=request_ids,
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonces=nonces,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout=None,
            priority=POC_REQUEST_PRIORITY,
        ))
    else:
        # Fallback: individual coroutines (legacy path).
        async def _run_one(rid: str, nonce: int) -> dict[str, Any]:
            return await engine_client.poc_compute(
                request_id=rid,
                block_hash=block_hash,
                block_height=block_height,
                public_key=public_key,
                nonce=nonce,
                seq_len=seq_len,
                k_dim=k_dim,
                timeout=None,
                priority=POC_REQUEST_PRIORITY,
            )

        poc_task = asyncio.ensure_future(
            asyncio.gather(
                *(_run_one(r, n) for r, n in zip(request_ids, nonces))
            )
        )

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
                    if (
                        local_timeout_count == 1
                        or local_timeout_count % 10 == 0
                    ):
                        logger.warning(
                            "PoC still pending after %.1fs (#%d); "
                            "engine likely busy",
                            timeout_sec,
                            local_timeout_count,
                        )
                    await asyncio.sleep(
                        POC_CHAT_BUSY_BACKOFF_SEC * 2
                    )
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

    return results


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
        results = await compute_artifact(
            engine_client=engine_client,
            nonces=nonces,
            block_hash=block_hash,
            block_height=block_height,
            public_key=public_key,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout_sec=timeout_sec,
        )
        return _build_artifacts(results)
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
    batch_size = _resolve_generation_batch_size(engine_client, config)
    pipeline_depth = 2

    logger.info(
        "Generation loop batch_size=%s pipeline_depth=%s ",
        batch_size,
        pipeline_depth,
    )

    compute_timeout = env.POC_RPC_TIMEOUT_MS / 1000.0

    def _make_task(ns: list[int]) -> asyncio.Task:
        return asyncio.create_task(compute_artifact(
            engine_client=engine_client,
            nonces=ns,
            block_hash=config.block_hash,
            block_height=config.block_height,
            public_key=config.public_key,
            seq_len=config.seq_len,
            k_dim=config.k_dim,
            timeout_sec=compute_timeout,
        ))

    in_flight_tasks: dict[asyncio.Task, list[int]] = {}

    for _ in range(pipeline_depth):
        task_nonces = nonce_iter.take(batch_size)
        in_flight_tasks[_make_task(task_nonces)] = task_nonces

    try:
        while not stop_event.is_set():
            if not in_flight_tasks:
                for _ in range(pipeline_depth):
                    task_nonces = nonce_iter.take(batch_size)
                    in_flight_tasks[_make_task(task_nonces)] = task_nonces

            done, _ = await asyncio.wait(
                in_flight_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished_task in done:
                task_nonces = in_flight_tasks.pop(finished_task)
                try:
                    results = finished_task.result()
                    timeout_count = 0
                    error_count = 0
                except TimeoutError:
                    timeout_count += 1
                    if timeout_count == 1 or timeout_count % 10 == 0:
                        logger.warning(
                            "Generation timed out (#%d), engine busy",
                            timeout_count,
                        )
                    await asyncio.sleep(
                        POC_CHAT_BUSY_BACKOFF_SEC * 2
                    )
                    in_flight_tasks[_make_task(task_nonces)] = task_nonces
                    continue
                except Exception:
                    error_count += 1
                    if error_count == 1 or error_count % 10 == 0:
                        logger.warning(
                            "Generation request failed (#%d), "
                            "backing off",
                            error_count,
                        )
                    await asyncio.sleep(
                        POC_CHAT_BUSY_BACKOFF_SEC * 2
                    )
                    in_flight_tasks[_make_task(task_nonces)] = task_nonces
                    continue

                if not stop_event.is_set():
                    next_nonces = nonce_iter.take(batch_size)
                    in_flight_tasks[_make_task(next_nonces)] = next_nonces

                if callback_sender:
                    artifacts = _build_callback_artifacts(results)
                else:
                    artifacts = []

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

                stats.total_processed += _count_result_nonces(results)

                current_time = time.time()
                if current_time - last_report_time >= 5.0:
                    window_sec = current_time - last_report_time
                    window_delta = (
                        stats.total_processed - last_report_total
                    )
                    rate = (
                        (window_delta / (window_sec / 60.0))
                        if window_sec > 0
                        else 0
                    )
                    logger.info(
                        "Generated: %d nonces (%.0f/min)",
                        stats.total_processed,
                        rate,
                    )
                    last_report_time = current_time
                    last_report_total = stats.total_processed

    except asyncio.CancelledError:
        for task in in_flight_tasks:
            task.cancel()
        elapsed_min = (time.time() - start_time) / 60
        logger.info(
            "Generation stopped: %d nonces in %.2fmin",
            stats.total_processed,
            elapsed_min,
        )
    except Exception as e:
        logger.exception("Generation crashed: %s", e)
        raise
