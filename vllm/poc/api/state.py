import asyncio
import contextlib
import time

from vllm.poc.protocol.config import PoCState
from vllm.poc.protocol.schemas import (
    PoCConfigSchema,
    PoCGenerationStatsSchema,
    StatusResponseSchema,
)
from vllm.poc.runtime.state import PoCAppTasks

_poc_tasks_typed: dict[int, PoCAppTasks] = {}


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
        tasks.stop_event.set()
        if tasks.callback_task is not None:
            tasks.callback_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.callback_task
        if tasks.gen_task is not None:
            tasks.gen_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tasks.gen_task
        if tasks.callback_sender is not None:
            tasks.callback_sender.clear()
