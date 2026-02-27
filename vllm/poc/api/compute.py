import asyncio
import uuid
from typing import Any

from vllm.poc.constants import POC_CHAT_BUSY_BACKOFF_SEC, POC_REQUEST_PRIORITY
from vllm.poc.protocol.types import Artifact
import vllm.poc.utils.env as env
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

    async def _run_all() -> list[dict[str, Any]]:
        return await asyncio.gather(*(_run_one(n) for n in nonces))

    poc_task = asyncio.create_task(_run_all())

    try:
        if timeout_sec is None:
            result = await poc_task
        else:
            local_timeout_count = 0
            while True:
                try:
                    result = await asyncio.wait_for(asyncio.shield(poc_task), timeout=timeout_sec)
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
            nonces,
        )
        raise

    all_nonces: list[int] = []
    all_vectors_b64: list[str] = []
    for r in result:
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
