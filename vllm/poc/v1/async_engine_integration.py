"""Async engine integration for PoC (Proof of Compute)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.v1.scheduler_params import PoCSchedulerParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind

# Prefill-only; we keep max_tokens=1 as a harmless placeholder.
# Immutable to avoid accidental mutation by call sites.
_POC_SAMPLING_PARAMS = SamplingParams(max_tokens=1, temperature=0.0)


@dataclass
class _WaiterEntry:
    """Tracks a PoC request awaiting completion."""

    future: asyncio.Future[dict[str, Any]]
    # Once tombstoned, late outputs must be ignored by output_handler.
    tombstoned: bool = False


def _now_wall() -> float:
    """Wall-clock timestamp used by scheduler arrival_time."""
    return time.time()


def _make_request(
    *,
    request_id: str,
    client_index: int,
    priority: int,
    seq_len: int,
    poc_params: PoCSchedulerParams,
) -> EngineCoreRequest:
    """Build a scheduler-native PoC request."""
    # Prompt token ids are dummy for PoC; embeddings are injected later.
    prompt_token_ids = [0] * int(seq_len)

    return EngineCoreRequest(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        mm_features=None,
        sampling_params=_POC_SAMPLING_PARAMS,
        pooling_params=None,
        eos_token_id=None,
        arrival_time=_now_wall(),
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
        client_index=client_index,
        priority=int(priority),
        kind=EngineCoreRequestKind.POC,
        poc_params=poc_params,
    )


async def poc_compute_impl(
    *,
    engine_core: Any,
    poc_waiters: MutableMapping[str, _WaiterEntry | asyncio.Future],
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
    """Submit one PoC nonce as a first-class scheduler request and await result.

    Safety properties:
    - Reject duplicate request_id (local process).
    - On timeout/cancel: tombstone waiter + best-effort abort.
    - Always cleanup waiter entry.

    The output_handler must resolve futures from `poc_waiters` and SHOULD ignore
    tombstoned entries to avoid late-result races.
    """
    if not request_id:
        raise ValueError("request_id must be non-empty")
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if int(k_dim) <= 0:
        raise ValueError(f"k_dim must be > 0, got {k_dim}")

    # Backward-compat: some call sites store raw Future; normalize to _WaiterEntry.
    if request_id in poc_waiters:
        raise ValueError(f"duplicate PoC request_id: {request_id}")

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    entry = _WaiterEntry(future=future)
    poc_waiters[request_id] = entry

    poc_params = PoCSchedulerParams(
        block_hash=block_hash,
        public_key=public_key,
        block_height=int(block_height),
        nonce=int(nonce),
        seq_len=int(seq_len),
        k_dim=int(k_dim),
    )

    request = _make_request(
        request_id=request_id,
        client_index=client_index,
        priority=priority,
        seq_len=seq_len,
        poc_params=poc_params,
    )

    # Submit to engine core first.
    await engine_core.add_request_async(request)

    try:
        if timeout is None:
            return await future
        return await asyncio.wait_for(future, timeout=float(timeout))

    except (asyncio.TimeoutError, asyncio.CancelledError):
        # Tombstone to prevent late output from resolving this future.
        entry.tombstoned = True
        # Best-effort: attempt to abort inside engine.
        with contextlib.suppress(Exception):
            await engine_core.abort_requests_async([request_id])
        # Re-raise to let caller apply retry/backoff policy.
        raise

    finally:
        # Always remove waiter mapping to prevent leaks.
        # If output arrives concurrently, output_handler should handle missing id.
        poc_waiters.pop(request_id, None)
