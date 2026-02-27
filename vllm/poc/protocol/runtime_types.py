"""Runtime-facing PoC data structures.

These dataclasses represent typed payloads used across the PoC runtime
(API routes, callbacks, and validation). They are *not* scheduler-native
(`vllm/poc/v1/*`).
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.poc.constants import DEFAULT_K_DIM


@dataclass
class PoCModelParams:
    """Model/shape parameters for runtime API requests."""

    model: str
    seq_len: int
    k_dim: int = DEFAULT_K_DIM

    @property
    def sequence_length(self) -> int:
        return int(self.seq_len)

    @property
    def projection_dimension(self) -> int:
        return int(self.k_dim)


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
