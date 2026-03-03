"""Scheduler integration helpers for PoC (Proof of Compute)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from vllm.v1.engine import (
    EngineCoreOutput,
    EngineCoreRequestKind,
    FinishReason,
)
from vllm.v1.request import RequestStatus

if TYPE_CHECKING:
    from vllm.v1.outputs import ModelRunnerOutput
    from vllm.v1.request import Request


def maybe_add_poc_request_id(request: Request, poc_req_ids: set[str]) -> None:
    """Add request id to the scheduled PoC set when the request is PoC."""
    if request.is_poc:
        poc_req_ids.add(request.request_id)


def build_poc_engine_core_output(
    *,
    request: Request,
    req_id: str,
    model_runner_output: ModelRunnerOutput,
    free_poc_request: Callable[[Request], None],
) -> EngineCoreOutput | None:
    """Build final EngineCoreOutput for a PoC request and free its state.

    Returns `None` for non-PoC requests.
    """
    if not request.is_poc:
        return None

    poc_result = None
    if model_runner_output.poc_results is not None:
        poc_result = model_runner_output.poc_results.get(req_id)

    if poc_result is None:
        request.status = RequestStatus.FINISHED_ERROR
        finish_reason = FinishReason.ERROR
        stop_reason: int | str | None = "poc_result_missing"
    else:
        request.status = RequestStatus.FINISHED_STOPPED
        finish_reason = FinishReason.STOP
        stop_reason = None

    free_poc_request(request)

    return EngineCoreOutput(
        request_id=req_id,
        new_token_ids=[],
        finish_reason=finish_reason,
        stop_reason=stop_reason,
        poc_result=poc_result,
        kind=EngineCoreRequestKind.POC,
        events=request.take_events(),
        trace_headers=request.trace_headers,
        num_cached_tokens=0,
    )