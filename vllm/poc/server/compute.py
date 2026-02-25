"""PoC nonce computation and generation loop.

PoC runs via direct worker RPC and bypasses the v1 scheduler request path.
This preserves the older execution model where a nonce batch is forwarded to
all workers with ``collective_rpc("execute_poc_forward")``.
"""

import asyncio
import time
from typing import Any

import vllm.poc.env as env
from vllm.poc._log import init_poc_logger
from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC
from vllm.poc.server.callbacks import CallbackSender
from vllm.poc.server.models import (
    Artifact,
    ArtifactBatchMeta,
    NonceIterator,
    PoCConfig,
    PoCGenerationStats,
)

logger = init_poc_logger(__name__)


def resolve_rpc_batch_size(requested_batch_size: int | None = None) -> int:
    """Resolve the direct PoC RPC batch size."""
    default_batch_size = env.POC_BATCH_SIZE_DEFAULT
    batch = requested_batch_size
    if batch is None or batch <= 0:
        batch = default_batch_size

    max_seqs = env.POC_MAX_NUM_SEQS
    if max_seqs > 0 and batch > max_seqs:
        logger.info(
            "Clamping batch_size=%s to POC_MAX_NUM_SEQS=%s",
            batch,
            max_seqs,
        )
        batch = max_seqs

    return int(batch)


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
        batch = default_batch_size
    elif requested_batch_size is None:
        batch = default_batch_size
    elif requested_batch_size <= 0:
        logger.warning(
            "Invalid init batch_size=%s; falling back to %s",
            requested_batch_size,
            default_batch_size,
        )
        batch = default_batch_size
    else:
        batch = requested_batch_size

    return resolve_rpc_batch_size(batch)


def _extract_hidden_size(engine_client: Any) -> int:
    model_config = engine_client.vllm_config.model_config
    return int(model_config.get_hidden_size())


def _build_artifacts(result: dict[str, Any] | None) -> list[Artifact]:
    if not result:
        return []

    nonces = result.get("nonces", [])
    vectors_b64 = result.get("vectors_b64", [])
    return [
        Artifact(nonce=int(nonce), vector_b64=str(vector_b64))
        for nonce, vector_b64 in zip(nonces, vectors_b64)
    ]


async def run_poc_rpc(
    engine_client,
    nonces: list[int],
    block_hash: str,
    public_key: str,
    seq_len: int,
    k_dim: int,
    timeout_sec: float | None = None,
) -> list[Artifact]:
    """Run PoC forward directly on workers via collective RPC."""

    if timeout_sec is None:
        timeout_sec = env.POC_GENERATE_CHUNK_TIMEOUT_SEC

    hidden_size = _extract_hidden_size(engine_client)
    results = await engine_client.collective_rpc(
        "execute_poc_forward",
        timeout=timeout_sec,
        args=(
            block_hash,
            public_key,
            nonces,
            seq_len,
            hidden_size,
            k_dim,
        ),
    )

    if isinstance(results, (list, tuple)):
        result = next((item for item in results if item is not None), None)
    else:
        result = results

    return _build_artifacts(result)


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
    """Run PoC forward via direct worker RPC and return artifacts."""

    if timeout_sec is None:
        timeout_sec = env.POC_GENERATE_CHUNK_TIMEOUT_SEC

    if not nonces:
        return []

    _ = block_height
    poc_task = asyncio.create_task(
        run_poc_rpc(
            engine_client=engine_client,
            nonces=nonces,
            block_hash=block_hash,
            public_key=public_key,
            seq_len=seq_len,
            k_dim=k_dim,
            timeout_sec=timeout_sec,
        )
    )

    try:
        if timeout_sec is None:
            return await poc_task
        else:
            local_timeout_count = 0
            while True:
                try:
                    return await asyncio.wait_for(
                        asyncio.shield(poc_task), timeout=timeout_sec
                    )
                except asyncio.TimeoutError:
                    if poc_task.done():
                        return await poc_task
                    local_timeout_count += 1
                    if local_timeout_count == 1 or local_timeout_count % 10 == 0:
                        logger.warning(
                            "PoC still pending after %.1fs (#%d); engine likely busy",
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
    batch_size = _resolve_generation_batch_size(config)
    pipeline_depth = 2

    logger.info(
        "Generation loop batch_size=%s pipeline_depth=%s",
        batch_size,
        pipeline_depth,
    )

    compute_timeout = env.POC_RPC_TIMEOUT_MS / 1000.0

    def _make_task(ns: list[int]) -> asyncio.Task:
        return asyncio.create_task(
            compute_artifact(
                engine_client=engine_client,
                nonces=ns,
                block_hash=config.block_hash,
                block_height=config.block_height,
                public_key=config.public_key,
                seq_len=config.seq_len,
                k_dim=config.k_dim,
                timeout_sec=compute_timeout,
            )
        )

    in_flight_tasks: dict[asyncio.Task, list[int]] = {}

    for _ in range(pipeline_depth):
        task_nonces = nonce_iter.take(batch_size)
        in_flight_tasks[_make_task(task_nonces)] = task_nonces

    try:
        while not stop_event.is_set():
            if not in_flight_tasks:
                task_nonces = nonce_iter.take(batch_size)
                in_flight_tasks[_make_task(task_nonces)] = task_nonces

            done, _ = await asyncio.wait(
                in_flight_tasks.keys(),
                return_when=asyncio.FIRST_COMPLETED,
            )

            for finished_task in done:
                task_nonces = in_flight_tasks.pop(finished_task)
                try:
                    artifacts = finished_task.result()
                    timeout_count = 0
                    error_count = 0
                except TimeoutError:
                    timeout_count += 1
                    if timeout_count == 1 or timeout_count % 10 == 0:
                        logger.warning(
                            "Generation timed out (#%d), engine busy",
                            timeout_count,
                        )
                    await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                    in_flight_tasks[_make_task(task_nonces)] = task_nonces
                    continue
                except Exception:
                    error_count += 1
                    if error_count == 1 or error_count % 10 == 0:
                        logger.warning(
                            "Generation request failed (#%d), backing off",
                            error_count,
                        )
                    await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                    in_flight_tasks[_make_task(task_nonces)] = task_nonces
                    continue

                if not stop_event.is_set():
                    next_nonces = nonce_iter.take(batch_size)
                    in_flight_tasks[_make_task(next_nonces)] = next_nonces

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
