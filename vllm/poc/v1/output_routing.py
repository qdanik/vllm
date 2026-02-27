# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import asyncio
from typing import Any


def resolve_poc_outputs(
    engine_core_outputs: list["EngineCoreOutput"],
    poc_waiters: dict[str, asyncio.Future[dict[str, Any]]],
) -> tuple[list["EngineCoreOutput"], int, int]:
    """Resolve PoC futures and return non-PoC outputs.

    Hardening goal: PoC outputs must never reach the normal OutputProcessor.
    This must hold even if the frontend timed out/cancelled and removed the
    waiter ("orphaned" PoC outputs).

    Returns:
      remaining_outputs: outputs safe for OutputProcessor
      resolved_count: number of futures resolved this call
      orphaned_count: PoC outputs without a matching waiter (dropped)
    """

    # Avoid import cycles at module load.
    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequestKind

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

        future = poc_waiters.pop(output.request_id, None)
        if future is None:
            orphaned += 1
            continue
        if future.done():
            continue

        if output.poc_result is None:
            future.set_exception(RuntimeError("PoC request failed"))
        else:
            future.set_result(output.poc_result)
        resolved += 1

    return remaining, resolved, orphaned
