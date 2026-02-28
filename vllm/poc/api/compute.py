import asyncio
import uuid
from typing import Any

import vllm.poc.env as env
from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC, POC_REQUEST_PRIORITY
from vllm.poc.protocol.runtime_types import Artifact
from vllm.poc.utils.poc_logger import init_poc_logger

logger = init_poc_logger(__name__)


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
        request_id = str(uuid.uuid4())
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

    max_inflight = int(getattr(env, "POC_BATCH_SIZE_DEFAULT", 32) or 32)
    if max_inflight <= 0:
        max_inflight = 1

    async def _run_chunk(chunk: list[int]) -> list[dict[str, Any]]:
        # Bound concurrency to avoid creating thousands of in-flight requests
        # and overwhelming the scheduler/frontend.
        semaphore = asyncio.Semaphore(max_inflight)

        async def _guarded_run(n: int) -> dict[str, Any]:
            async with semaphore:
                return await _run_one(n)

        return await asyncio.gather(*(_guarded_run(n) for n in chunk))

    def _iter_chunks(items: list[int], chunk_size: int) -> list[list[int]]:
        if chunk_size <= 0:
            chunk_size = 1
        return [items[i : i + chunk_size] for i in range(0, len(items), chunk_size)]

    # Submit in bounded chunks. Note: within each chunk, requests are still
    # independent so the scheduler can auto-batch them.
    chunks = _iter_chunks(nonces, max_inflight)
    results: list[dict[str, Any]] = []

    for chunk in chunks:
        poc_task = asyncio.create_task(_run_chunk(chunk))

        try:
            if timeout_sec is None:
                result = await poc_task
            else:
                local_timeout_count = 0
                while True:
                    try:
                        result = await asyncio.wait_for(
                            asyncio.shield(poc_task), timeout=timeout_sec
                        )
                        break
                    except asyncio.TimeoutError:
                        if poc_task.done():
                            result = await poc_task
                            break
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
                chunk,
            )
            raise

        results.extend(result)

    all_nonces: list[int] = []
    all_vectors_b64: list[str] = []
    for r in results:
        if not r:
            continue
        all_nonces.extend(r.get("nonces", []))
        all_vectors_b64.extend(r.get("vectors_b64", []))

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
            engine_client,
            nonces,
            block_hash,
            block_height,
            public_key,
            seq_len,
            k_dim,
            timeout_sec,
        )
    except TimeoutError as e:
        raise RuntimeError(f"Timeout after {timeout_sec}s") from e
