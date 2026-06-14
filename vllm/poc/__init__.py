# Apply PoC engine patch for vLLM 0.15.1 V1 engine
from . import engine_patch

from .config import PoCConfig, PoCState
from .data import (
    PoCParams,
    Artifact,
    Encoding,
    ArtifactBatch,
    ValidationResult,
    encode_vector,
    decode_vector,
    is_mismatch,
    fraud_test,
    compare_artifacts,
)
from .manager import PoCManager
from .routes import router as poc_router
from .layer_hooks import LayerHouseholderHook
from .fingerprint_hooks import RoutingFingerprintHook
from .fingerprint import (
    CapturedDecision,
    NonceFingerprint,
    to_dump_jsonl,
    write_meta_sidecar,
    pick_routing_sites,
    pick_logit_positions,
    router_logits_to_decisions,
    capture_seeded_logits,
)

__all__ = [
    "PoCConfig",
    "PoCState",
    "PoCParams",
    "Artifact",
    "Encoding",
    "ArtifactBatch",
    "ValidationResult",
    "encode_vector",
    "decode_vector",
    "is_mismatch",
    "fraud_test",
    "compare_artifacts",
    "PoCManager",
    "poc_router",
    "LayerHouseholderHook",
    "RoutingFingerprintHook",
    "CapturedDecision",
    "NonceFingerprint",
    "to_dump_jsonl",
    "write_meta_sidecar",
    "pick_routing_sites",
    "pick_logit_positions",
    "router_logits_to_decisions",
    "capture_seeded_logits",
]
