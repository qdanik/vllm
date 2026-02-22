"""PoC protocol types and configuration.

Public API for request/response structures, artifacts, and config.
"""

from vllm.poc.protocol.config import PoCConfig, PoCState
from vllm.poc.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.protocol.types import (
    Artifact,
    ArtifactBatchMeta,
    ArtifactValidationStats,
    Encoding,
    PoCParams,
)

__all__ = [
    "PoCState",
    "PoCConfig",
    "DEFAULT_K_DIM",
    "DEFAULT_DIST_THRESHOLD",
    "DEFAULT_P_MISMATCH",
    "DEFAULT_FRAUD_THRESHOLD",
    "PoCParams",
    "Artifact",
    "ArtifactBatchMeta",
    "ArtifactValidationStats",
    "Encoding",
]
