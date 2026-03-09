"""PoC application state management and request helpers.

Combines the former ``api/state`` and ``api/helpers`` modules.
"""

import asyncio
import contextlib
import time
import uuid

from fastapi import HTTPException, Request

from vllm.poc.server.models import (
    PoCAppTasks,
    PoCParamsModel,
    PoCState,
)
from vllm.poc.server.schemas import (
    PoCConfigSchema,
    PoCGenerationStatsSchema,
    StatusResponseSchema,
)

_poc_tasks_typed: dict[int, PoCAppTasks] = {}
_sprint_tasks_typed: dict[int, PoCAppTasks] = {}


def is_sprint_active(app_id: int) -> bool:
    """Return True when a sprint loop is running for *app_id*."""
    tasks = _sprint_tasks_typed.get(app_id)
    if tasks is None:
        return False
    gen_task = tasks.gen_task
    return gen_task is not None and not gen_task.done()


def get_api_sprint_status(app_id: int) -> StatusResponseSchema:
    tasks = _sprint_tasks_typed.get(app_id)

    if tasks is None or not is_sprint_active(app_id):
        return StatusResponseSchema(status=PoCState.IDLE, config=None, stats=None)

    config = tasks.config
    stats = tasks.stats
    start_time = stats.start_time
    total_processed = stats.total_processed
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0

    return StatusResponseSchema(
        status=PoCState.GENERATING,
        config=PoCConfigSchema.from_config(config),
        stats=PoCGenerationStatsSchema(
            total_processed=total_processed,
            nonces_per_second=nonces_per_second,
        ),
    )


async def cancel_sprint_tasks(app_id: int) -> None:
    """Cancel and clean up an active sprint loop."""
    tasks = _sprint_tasks_typed.pop(app_id, None)
    if tasks is not None:
        # Stop generation first.
        tasks.stop_event.set()
        if tasks.gen_task is not None:
            tasks.gen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.gen_task
        # Let the callback sender finish: it will flush remaining artifacts
        # before exiting the loop (triggered by stop_event being set).
        if tasks.callback_task is not None:
            try:
                await asyncio.wait_for(tasks.callback_task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                tasks.callback_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tasks.callback_task


def is_generation_active(app_id: int) -> bool:
    tasks = _poc_tasks_typed.get(app_id)
    if tasks is None:
        return False
    gen_task = tasks.gen_task
    return gen_task is not None and not gen_task.done()


def get_api_status(app_id: int) -> StatusResponseSchema:
    tasks = _poc_tasks_typed.get(app_id)

    if tasks is None or not is_generation_active(app_id):
        return StatusResponseSchema(status=PoCState.IDLE, config=None, stats=None)

    config = tasks.config
    stats = tasks.stats
    start_time = stats.start_time
    total_processed = stats.total_processed
    elapsed = time.time() - start_time if start_time > 0 else 0
    nonces_per_second = total_processed / elapsed if elapsed > 0 else 0

    return StatusResponseSchema(
        status=PoCState.GENERATING,
        config=PoCConfigSchema.from_config(config),
        stats=PoCGenerationStatsSchema(
            total_processed=total_processed,
            nonces_per_second=nonces_per_second,
        ),
    )


async def cancel_poc_tasks(app_id: int):
    tasks = _poc_tasks_typed.pop(app_id, None)
    if tasks is not None:
        # Stop generation first.
        tasks.stop_event.set()
        if tasks.gen_task is not None:
            tasks.gen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.gen_task
        # Let the callback sender finish: it will flush remaining artifacts
        # before exiting the loop (triggered by stop_event being set).
        if tasks.callback_task is not None:
            try:
                await asyncio.wait_for(tasks.callback_task, timeout=15.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                tasks.callback_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await tasks.callback_task


async def get_engine_client(request: Request):
    engine_client = getattr(request.app.state, "engine_client", None)
    if engine_client is None:
        raise HTTPException(status_code=503, detail="Engine not available")
    return engine_client


def check_params_match(request: Request, params: PoCParamsModel):
    """Check params match deployed config. Raises 409 on mismatch."""
    serving_models = getattr(request.app.state, "openai_serving_models", None)
    if serving_models and hasattr(serving_models, "base_model_paths"):
        base_paths = serving_models.base_model_paths
        if base_paths:
            model_path = base_paths[0].model_path
            served_names = [p.name for p in base_paths]
            valid_models = {model_path} | set(served_names)
            if params.model not in valid_models:
                raise HTTPException(
                    status_code=409,
                    detail={
                        "error": "params mismatch",
                        "requested": {
                            "model": params.model,
                            "seq_len": params.seq_len,
                            "k_dim": params.k_dim,
                        },
                        "deployed": {
                            "model": list(valid_models),
                            "seq_len": None,
                            "k_dim": None,
                        },
                    },
                )

    deployed = getattr(request.app.state, "poc_deployed", None)
    if deployed:
        mismatches = []
        if deployed.get("model") and params.model != deployed["model"]:
            mismatches.append("model")
        if deployed.get("seq_len") and params.seq_len != deployed["seq_len"]:
            mismatches.append("seq_len")
        if deployed.get("k_dim") and params.k_dim != deployed["k_dim"]:
            mismatches.append("k_dim")

        if mismatches:
            raise HTTPException(
                status_code=409,
                detail={
                    "error": "params mismatch",
                    "fields": mismatches,
                    "requested": {
                        "model": params.model,
                        "seq_len": params.seq_len,
                        "k_dim": params.k_dim,
                    },
                    "deployed": deployed,
                },
            )


def generate_request_id() -> str:
    return str(uuid.uuid4())
