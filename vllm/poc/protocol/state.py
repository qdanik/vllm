"""Typed runtime state for PoC API.

This module avoids using anonymous dicts for mutable in-memory state shared
between API handlers.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import TYPE_CHECKING

from vllm.poc.protocol.config import PoCConfig

if TYPE_CHECKING:
    from vllm.poc.protocol.callbacks import CallbackSender


@dataclass
class PoCGenerationStats:
    start_time: float = 0.0
    total_processed: int = 0


@dataclass
class PoCAppTasks:
    gen_task: asyncio.Task[None] | None
    callback_task: asyncio.Task[None] | None
    callback_sender: CallbackSender | None
    stop_event: asyncio.Event
    config: PoCConfig
    stats: PoCGenerationStats
