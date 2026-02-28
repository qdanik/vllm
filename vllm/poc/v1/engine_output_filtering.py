# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from collections.abc import MutableMapping
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.poc.v1.async_engine_integration import PoCWaiterEntry
    from vllm.v1.engine import EngineCoreOutput


def resolve_poc_outputs(
    engine_core_outputs: list[EngineCoreOutput],
    poc_waiters: MutableMapping[str, PoCWaiterEntry],
) -> tuple[list[EngineCoreOutput], int, int]:
    """Resolve PoC futures and return non-PoC outputs.

    Hardening goal: PoC outputs must never reach the normal OutputProcessor.
    This must hold even if the frontend timed out/cancelled and removed the
    waiter ("orphaned" PoC outputs).

    Returns:
      remaining: outputs safe for OutputProcessor
      resolved: number of futures resolved this call
      orphaned: PoC outputs without a matching waiter (dropped)
    """

    # Avoid import cycles at module load.
    from vllm.v1.engine import EngineCoreRequestKind

    remaining: list[EngineCoreOutput] = []
    resolved = 0
    orphaned = 0

    for output in engine_core_outputs:
        is_poc = (output.kind == EngineCoreRequestKind.POC) or (
            output.poc_result is not None
        )
        if not is_poc:
            remaining.append(output)
            continue

        entry = poc_waiters.pop(output.request_id, None)
        if entry is None:
            orphaned += 1
            continue

        # If we timed out/cancelled earlier, drop late results.
        if entry.tombstoned:
            orphaned += 1
            continue

        future = entry.future

        if future.done():
            continue

        if output.poc_result is None:
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
            detail = f"finish_reason={finish_reason} stop_reason={stop_reason}"
            future.set_exception(RuntimeError(f"PoC request failed ({detail})"))
        else:
            future.set_result(output.poc_result)
        resolved += 1

    return remaining, resolved, orphaned
