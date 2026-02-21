"""PoC protocol data types and request/response structures.

These dataclasses represent the typed, structured payloads used across the PoC
runtime (API routes, callbacks, validation).
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.poc.protocol.constants import DEFAULT_K_DIM


@dataclass
class PoCParams:
    """Strict params for PoC requests - exactly 3 fields."""

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
