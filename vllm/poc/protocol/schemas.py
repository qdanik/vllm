"""Pydantic schemas for PoC runtime JSON boundaries.

Internal logic should operate on typed objects; these schemas define the JSON
shape for FastAPI responses and HTTP callback payloads.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer

from vllm.poc.protocol.config import PoCConfig, PoCState
from vllm.poc.protocol.enums import ApiStatus, GenerateResultStatus, GenerateStatus


class EncodingSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    dtype: str = "f16"
    k_dim: int
    endian: str = "le"


class ArtifactSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    nonce: int
    vector_b64: str


class ArtifactBatchSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    public_key: str
    block_hash: str
    block_height: int
    node_id: int
    artifacts: list[ArtifactSchema]
    encoding: EncodingSchema


class GeneratedCallbackPayloadSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    artifacts: list[ArtifactSchema]
    encoding: EncodingSchema


class ValidatedCallbackPayloadSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    request_id: str
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    n_total: int
    n_mismatch: int
    mismatch_nonces: list[int]
    p_value: float
    fraud_detected: bool


class PowStatusSchema(BaseModel):
    status: PoCState


def _default_pow_status_generating() -> PowStatusSchema:
    return PowStatusSchema(status=PoCState.GENERATING)


def _default_pow_status_stopped() -> PowStatusSchema:
    return PowStatusSchema(status=PoCState.STOPPED)


class InitGenerateResponseSchema(BaseModel):
    status: ApiStatus = ApiStatus.OK
    pow_status: PowStatusSchema = Field(default_factory=_default_pow_status_generating)


class StopResponseSchema(BaseModel):
    status: ApiStatus = ApiStatus.OK
    pow_status: PowStatusSchema = Field(default_factory=_default_pow_status_stopped)


class PoCGenerationStatsSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    total_processed: int
    nonces_per_second: float


class PoCConfigSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int
    n_groups: int
    seq_len: int
    k_dim: int
    batch_size: int

    @classmethod
    def from_config(cls, config: PoCConfig) -> PoCConfigSchema:
        return cls(
            block_hash=config.block_hash,
            block_height=config.block_height,
            public_key=config.public_key,
            node_id=config.node_id,
            node_count=config.node_count,
            group_id=config.group_id,
            n_groups=config.n_groups,
            seq_len=config.seq_len,
            k_dim=config.k_dim,
            batch_size=config.batch_size,
        )


class StatusResponseSchema(BaseModel):
    status: PoCState
    config: PoCConfigSchema | None
    stats: PoCGenerationStatsSchema | None


class GenerateQueuedResponseSchema(BaseModel):
    status: Literal[GenerateStatus.QUEUED] = GenerateStatus.QUEUED
    request_id: str
    queued_count: int


class GenerateCompletedResponseSchema(BaseModel):
    status: Literal[GenerateStatus.COMPLETED] = GenerateStatus.COMPLETED
    request_id: str
    artifacts: list[ArtifactSchema]
    encoding: EncodingSchema


class GenerateValidatedCompletedResponseSchema(BaseModel):
    status: Literal[GenerateStatus.COMPLETED] = GenerateStatus.COMPLETED
    request_id: str
    n_total: int
    n_mismatch: int
    mismatch_nonces: list[int]
    p_value: float
    fraud_detected: bool


GenerateResponseSchema = (
    GenerateQueuedResponseSchema
    | GenerateCompletedResponseSchema
    | GenerateValidatedCompletedResponseSchema
)


class GetGenerateResultResponseSchema(BaseModel):
    """Response for GET /generate/{request_id}.

    Wire-compat: keeps legacy shape where completed payload is flattened into
    the top-level JSON object.
    """

    status: GenerateResultStatus
    request_id: str
    payload: (
        GenerateCompletedResponseSchema
        | GenerateValidatedCompletedResponseSchema
        | None
    ) = None
    error: str | None = None

    @model_serializer(mode="wrap")
    def _ser(self, handler):
        data: dict[str, Any] = handler(self)
        payload = data.pop("payload", None)
        if payload:
            # Merge completed fields to top-level for backward compatibility.
            data.update(payload)
        # Drop nulls
        return {k: v for k, v in data.items() if v is not None}
