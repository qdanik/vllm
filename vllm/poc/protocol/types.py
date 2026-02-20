"""PoC protocol data types and request/response structures.

Dataclasses for API payloads, artifacts, and validation results.
"""

from dataclasses import dataclass


@dataclass
class PoCParams:
    """Strict params for PoC requests - exactly 3 fields."""

    model: str
    seq_len: int
    k_dim: int = 12  # DEFAULT_K_DIM


@dataclass
class Artifact:
    """Single nonce artifact with base64-encoded vector."""

    nonce: int
    vector_b64: str


@dataclass
class Encoding:
    """Metadata for vector encoding."""

    dtype: str = "f16"
    k_dim: int = 12  # DEFAULT_K_DIM
    endian: str = "le"


@dataclass
class ArtifactBatch:
    """Batch of artifacts for callback payloads."""

    public_key: str
    block_hash: str
    block_height: int
    node_id: int
    artifacts: list[Artifact]
    encoding: Encoding


@dataclass
class ValidationResult:
    """Result of artifact validation."""

    public_key: str
    block_hash: str
    block_height: int
    node_id: int
    nonces: list[int]
    n_total: int
    n_mismatch: int
    mismatch_nonces: list[int]
    fraud_threshold: float = 0.05  # DEFAULT_FRAUD_THRESHOLD
    p_value: float | None = None
    fraud_detected: bool | None = None
