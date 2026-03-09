"""Sprint mode: maximum-throughput PoC generation via collective_rpc (no scheduler)."""

from __future__ import annotations

import asyncio
import base64
import time
from typing import Any

import numpy as np

import vllm.poc.env as env
from vllm.poc._log import init_poc_logger
from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC
from vllm.poc.server.compute import _max_schedulable_reqs_for_seq_len
from vllm.poc.server.models import (
    Artifact,
    ArtifactBatchMeta,
    NonceIterator,
    PoCConfig,
    PoCGenerationStats,
)
from vllm.poc.server.sprint_runner import execute_sprint_forward_multi_batch

# Pass callable directly; collective_rpc serializes it once.
_SPRINT_METHOD = execute_sprint_forward_multi_batch

logger = init_poc_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _encode_vectors_batch(
    vectors: np.ndarray, nonces: list[int]
) -> list[Artifact]:
    """Encode FP16 numpy rows as base64 Artifact objects (single allocation)."""
    if not vectors.flags["C_CONTIGUOUS"]:
        vectors = np.ascontiguousarray(vectors)
    row_bytes = vectors.strides[0]
    raw = vectors.tobytes()
    return [
        Artifact(
            nonce=nonce,
            vector_b64=base64.b64encode(
                raw[i * row_bytes : (i + 1) * row_bytes]
            ).decode("ascii"),
        )
        for i, nonce in enumerate(nonces)
    ]


def _get_hidden_size(engine_client: Any) -> int | None:
    try:
        cfg = getattr(engine_client, "vllm_config", None)
        model_cfg = getattr(cfg, "model_config", None) if cfg else None
        if model_cfg is not None:
            return int(model_cfg.get_hidden_size())
    except Exception:
        pass
    return None


def _resolve_batch_size(config: PoCConfig) -> int:
    default = max(env.POC_BATCH_SIZE_DEFAULT, 1)
    batch = (
        config.batch_size
        if (config.batch_size is not None and config.batch_size > 0)
        else default
    )
    max_schedulable = _max_schedulable_reqs_for_seq_len(config.seq_len)
    if batch > max_schedulable:
        logger.info(
            "[0.9.1] Clamping batch_size=%d to %d "
            "(seq_len=%d, max_seqs=%d, max_batched_tokens=%d)",
            batch, max_schedulable, config.seq_len,
            env.POC_MAX_NUM_SEQS, env.POC_MAX_NUM_BATCHED_TOKENS,
        )
        batch = max_schedulable
    return batch


def _resolve_pipeline_depth() -> int:
    depth = env.POC_GENERATION_PIPELINE_DEPTH
    return max(depth, 1) if depth > 0 else 4


# ---------------------------------------------------------------------------
# Sprint generation loop
# ---------------------------------------------------------------------------


async def sprint_generation_loop(
    engine_client: Any,
    stop_event: asyncio.Event,
    callback_sender: Any | None,
    config: PoCConfig,
    stats: PoCGenerationStats,
) -> None:
    """Maximum-throughput PoC generation via collective_rpc_async."""
    hidden_size = _get_hidden_size(engine_client)
    if not hidden_size:
        logger.error("[0.9.1] Cannot determine model hidden_size. Aborting.")
        return

    if not hasattr(engine_client, "collective_rpc"):
        logger.error(
            "[0.9.1] engine_client.collective_rpc not available. "
            "Sprint requires AsyncLLM (multi-process) mode."
        )
        return

    batch_size = _resolve_batch_size(config)
    pipeline_depth = _resolve_pipeline_depth()
    nonces_per_rpc = batch_size
    rpc_timeout = env.POC_RPC_TIMEOUT_MS / 1000.0

    logger.info(
        "[0.9.1] Starting: batch_size=%d nonces/RPC, "
        "pipeline_depth=%d, hidden_size=%d (node %s/%s, group %s/%s)",
        nonces_per_rpc,
        pipeline_depth,
        hidden_size,
        config.node_id,
        config.node_count,
        config.group_id,
        config.n_groups,
    )

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
    timeout_count = 0
    error_count = 0

    def _make_task(ns: list[int]) -> asyncio.Task:
        return asyncio.create_task(
            engine_client.collective_rpc(
                _SPRINT_METHOD,
                timeout=rpc_timeout,
                args=(
                    config.block_hash,
                    config.public_key,
                    ns,
                    batch_size,
                    config.seq_len,
                    hidden_size,
                    config.k_dim,
                ),
            )
        )

    # Seed the pipeline.
    in_flight: dict[asyncio.Task, list[int]] = {}
    for _ in range(pipeline_depth):
        ns = nonce_iter.take(nonces_per_rpc)
        in_flight[_make_task(ns)] = ns

    try:
        while not stop_event.is_set():
            if not in_flight:
                ns = nonce_iter.take(nonces_per_rpc)
                in_flight[_make_task(ns)] = ns

            done, _ = await asyncio.wait(
                in_flight.keys(), return_when=asyncio.FIRST_COMPLETED
            )

            for finished in done:
                task_nonces = in_flight.pop(finished)
                exc = finished.exception()

                if exc is not None:
                    is_timeout = isinstance(exc, (TimeoutError, asyncio.TimeoutError))
                    if is_timeout:
                        timeout_count += 1
                        if timeout_count == 1 or timeout_count % 10 == 0:
                            logger.warning(
                                "[Sprint] RPC timeout (#%d), engine busy",
                                timeout_count,
                            )
                    else:
                        error_count += 1
                        if error_count == 1 or error_count % 10 == 0:
                            logger.warning(
                                "[Sprint] RPC error (#%d): %s",
                                error_count,
                                exc,
                            )
                    await asyncio.sleep(POC_CHAT_BUSY_BACKOFF_SEC * 2)
                    in_flight[_make_task(task_nonces)] = task_nonces
                    continue

                timeout_count = 0
                error_count = 0

                # collective_rpc_async returns list[result_per_worker];
                # only the last PP rank returns a non-None dict.
                worker_results: list = finished.result()
                result = next(
                    (r for r in worker_results if r is not None), None
                )

                if not stop_event.is_set():
                    ns = nonce_iter.take(nonces_per_rpc)
                    in_flight[_make_task(ns)] = ns

                if result is not None:
                    nonces: list[int] = result["nonces"]
                    vectors: np.ndarray = np.frombuffer(
                        result["vectors_bytes"], dtype=np.float16
                    ).reshape(result["n"], result["k_dim"])
                    artifacts = _encode_vectors_batch(vectors, nonces)

                    if callback_sender is not None:
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

                now = time.time()
                if now - last_report_time >= 5.0:
                    window = now - last_report_time
                    delta = stats.total_processed - last_report_total
                    rate = (delta / (window / 60.0)) if window > 0 else 0.0
                    logger.info(
                        "[0.9.1] %d nonces generated (%.0f/min)",
                        stats.total_processed,
                        rate,
                    )
                    last_report_time = now
                    last_report_total = stats.total_processed

    except asyncio.CancelledError:
        for task in in_flight:
            task.cancel()
        elapsed = (time.time() - start_time) / 60.0
        logger.info(
            "[0.9.1] Stopped: %d nonces in %.2f min",
            stats.total_processed,
            elapsed,
        )
    except Exception as e:
        logger.exception("[0.9.1] Loop crashed: %s", e)
        raise

