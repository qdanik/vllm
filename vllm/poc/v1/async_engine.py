"""Async engine integration for PoC (Proof of Compute)."""

import asyncio
import contextlib
import time
from typing import Any

from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.v1.params import PoCParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind


async def poc_compute_impl(
    *,
    engine_core,
    poc_waiters: dict[str, asyncio.Future],
    request_id: str,
    block_hash: str,
    public_key: str,
    block_height: int,
    nonce: int,
    seq_len: int,
    k_dim: int,
    client_index: int,
    timeout: float | None = None,
    priority: int = POC_REQUEST_PRIORITY,
) -> dict[str, Any]:
    """Submit one PoC nonce as a first-class scheduler request and await the result.

    Args:
        engine_core: The engine core instance.
        poc_waiters: Dictionary to store pending PoC futures.
        request_id: Unique ID for this PoC request.
        block_hash: Blockchain block hash.
        public_key: Public key for verification.
        block_height: Block height.
        nonce: The nonce to compute.
        seq_len: Embedding sequence length.
        k_dim: Embedding dimension.
        client_index: Client index for the request.
        timeout: Optional timeout in seconds.
        priority: Request priority.

    Returns:
        Dictionary with PoC computation results.
    """
    if request_id in poc_waiters:
        raise ValueError(f"duplicate PoC request_id: {request_id}")

    fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
    poc_waiters[request_id] = fut

    sp = SamplingParams(max_tokens=1, temperature=0.0)

    req = EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=[0] * seq_len,
        mm_features=None,
        sampling_params=sp,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=time.time(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=client_index,
        priority=priority,
        kind=EngineCoreRequestKind.POC,
        poc_params=PoCParams(
            block_hash=block_hash,
            public_key=public_key,
            block_height=block_height,
            nonce=nonce,
            seq_len=seq_len,
            k_dim=k_dim,
        ),
    )

    await engine_core.add_request_async(req)

    try:
        if timeout is None:
            return await fut
        return await asyncio.wait_for(fut, timeout=timeout)
    except (asyncio.TimeoutError, asyncio.CancelledError):
        # Best-effort abort.
        poc_waiters.pop(request_id, None)
        with contextlib.suppress(Exception):
            await engine_core.abort_requests_async([request_id])
        raise
