"""Shared data types for the PoC server layer.

Consolidates configuration, runtime data structures, enums, and state
that were previously spread across ``protocol/config``, ``protocol/
runtime_types``, ``protocol/status_enums``, ``protocol/state``, and
``api/models``.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from vllm.poc.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)

if TYPE_CHECKING:
    from vllm.poc.server.callbacks import CallbackSender


# ── Enums ───────────────────────────────────────────────────────────────────


class PoCState(Enum):
    """State of PoC generation process."""

    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"


class ApiStatus(str, Enum):
    OK = "OK"


class GenerateStatus(str, Enum):
    QUEUED = "queued"
    COMPLETED = "completed"


class GenerateResultStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CallbackPath(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"


# ── Configuration ───────────────────────────────────────────────────────────


@dataclass
class PoCConfig:
    """Configuration for a PoC generation round."""

    block_hash: str
    block_height: int
    public_key: str
    node_id: int = 0
    node_count: int = 1
    seq_len: int = 256
    k_dim: int = DEFAULT_K_DIM
    callback_url: str | None = None
    group_id: int = 0
    n_groups: int = 1


# ── Runtime data structures ─────────────────────────────────────────────────


@dataclass
class PoCModelParams:
    """Model/shape parameters for runtime API requests."""

    model: str
    seq_len: int
    k_dim: int = DEFAULT_K_DIM


@dataclass
class Artifact:
    """Single nonce artifact with base64-encoded vector."""

    nonce: int
    vector_b64: str


@dataclass
class Encoding:
    """Metadata for vector encoding."""

    dtype: str = "f16"
    k_dim: int = DEFAULT_K_DIM
    endian: str = "le"


@dataclass
class ArtifactBatchMeta:
    """Metadata fields shared across callback payloads."""

    public_key: str
    block_hash: str
    block_height: int
    node_id: int


@dataclass
class ArtifactValidationStats:
    """Summary stats produced by artifact validation."""

    n_total: int
    n_mismatch: int
    mismatch_nonces: list[int]
    p_value: float
    fraud_detected: bool


# ── API request models (Pydantic) ──────────────────────────────────────────


class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = DEFAULT_K_DIM


class ArtifactSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    nonce: int
    vector_b64: str


class ValidationModel(BaseModel):
    artifacts: list[ArtifactSchema]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    params: PoCParamsModel
    url: str | None = None


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: list[int]
    params: PoCParamsModel
    wait: bool = False
    url: str | None = None
    validation: ValidationModel | None = None
    stat_test: StatTestModel | None = None


# ── Nonce iterator ──────────────────────────────────────────────────────────


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support."""

    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> list[int]:
        return [next(self) for _ in range(n)]


# ── App state ───────────────────────────────────────────────────────────────


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
