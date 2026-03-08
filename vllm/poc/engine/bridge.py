"""Async engine integration for PoC (Proof of Compute)."""

from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import MutableMapping
from dataclasses import dataclass
from typing import Any

from vllm.poc.constants import POC_REQUEST_PRIORITY
from vllm.poc.engine.params import PoCSchedulerParams
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreRequest, EngineCoreRequestKind

_ENGINE_CORE_REQUEST_FIELDS = frozenset(
    getattr(EngineCoreRequest, "__struct_fields__", ())
)

# Prefill-only; we keep max_tokens=1 as a harmless placeholder.
# Immutable to avoid accidental mutation by call sites.
_POC_SAMPLING_PARAMS = SamplingParams(max_tokens=1, temperature=0.0)


@dataclass
class PoCWaiterEntry:
    """Tracks a PoC request awaiting completion.

    We keep a tombstone flag so timeouts/cancels can safely ignore late outputs
    (race between output delivery and waiter cleanup).
    """

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

    request_kwargs: dict[str, Any] = {
        "request_id": request_id,
        "prompt_token_ids": prompt_token_ids,
        "mm_features": None,
        "sampling_params": _POC_SAMPLING_PARAMS,
        "pooling_params": None,
        "arrival_time": _now_wall(),
        "lora_request": None,
        "cache_salt": None,
        "data_parallel_rank": None,
        "client_index": client_index,
        "priority": int(priority),
        "kind": EngineCoreRequestKind.POC,
        "poc_params": poc_params,
    }

    # Compatibility guard across vLLM versions where EngineCoreRequest fields
    # evolve (e.g. eos_token_id moved under sampling_params).
    if _ENGINE_CORE_REQUEST_FIELDS:
        request_kwargs = {
            key: value
            for key, value in request_kwargs.items()
            if key in _ENGINE_CORE_REQUEST_FIELDS
        }

    return EngineCoreRequest(**request_kwargs)


async def poc_compute_impl(
    *,
    engine_core: Any,
    poc_waiters: MutableMapping[str, PoCWaiterEntry],
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

    if request_id in poc_waiters:
        raise ValueError(f"duplicate PoC request_id: {request_id}")

    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    entry = PoCWaiterEntry(future=future)
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


async def poc_compute_batch_impl(
    *,
    engine_core: Any,
    poc_waiters: MutableMapping[str, PoCWaiterEntry],
    request_ids: list[str],
    block_hash: str,
    public_key: str,
    block_height: int,
    nonces: list[int],
    seq_len: int,
    k_dim: int,
    client_index: int,
    timeout: float | None = None,
    priority: int = POC_REQUEST_PRIORITY,
) -> list[dict[str, Any]]:
    """Submit a batch of PoC nonces with minimal per-request overhead.

    Compared to calling poc_compute_impl N times via asyncio.gather,
    this avoids N coroutine creations and interleaved yields:
    1. Pre-build all requests and futures (pure CPU, no IPC).
    2. Submit all requests in a tight loop.
    3. Await all futures with a single gather.
    """
    if not nonces:
        return []
    n = len(nonces)
    if len(request_ids) != n:
        raise ValueError(
            "request_ids and nonces must have same length"
        )
    if int(seq_len) <= 0:
        raise ValueError(f"seq_len must be > 0, got {seq_len}")
    if int(k_dim) <= 0:
        raise ValueError(f"k_dim must be > 0, got {k_dim}")

    loop = asyncio.get_running_loop()

    # --- Phase 1: build all requests + futures (no IPC) ---
    futures: list[asyncio.Future[dict[str, Any]]] = []
    entries: list[PoCWaiterEntry] = []
    requests: list[Any] = []  # EngineCoreRequest
    for rid, nonce in zip(request_ids, nonces):
        if rid in poc_waiters:
            raise ValueError(f"duplicate PoC request_id: {rid}")
        fut: asyncio.Future[dict[str, Any]] = loop.create_future()
        entry = PoCWaiterEntry(future=fut)
        poc_waiters[rid] = entry
        futures.append(fut)
        entries.append(entry)
        requests.append(_make_request(
            request_id=rid,
            client_index=client_index,
            priority=priority,
            seq_len=seq_len,
            poc_params=PoCSchedulerParams(
                block_hash=block_hash,
                public_key=public_key,
                block_height=int(block_height),
                nonce=int(nonce),
                seq_len=int(seq_len),
                k_dim=int(k_dim),
            ),
        ))

    # --- Phase 2: submit all requests in ONE IPC frame ---
    batch_fn = getattr(engine_core, "add_requests_batch_async", None)
    if batch_fn is not None:
        await batch_fn(requests)
    else:
        # Fallback: individual IPC calls.
        for req in requests:
            await engine_core.add_request_async(req)

    # --- Phase 3: await all futures ---
    try:
        if timeout is None:
            return list(await asyncio.gather(*futures))
        return list(
            await asyncio.wait_for(
                asyncio.gather(*futures),
                timeout=float(timeout),
            )
        )
    except (asyncio.TimeoutError, asyncio.CancelledError):
        for entry in entries:
            entry.tombstoned = True
        with contextlib.suppress(Exception):
            await engine_core.abort_requests_async(request_ids)
        raise
    finally:
        for rid in request_ids:
            poc_waiters.pop(rid, None)
