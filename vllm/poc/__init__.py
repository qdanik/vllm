"""vLLM Proof-of-Compute (PoC) Module

This module provides consensus-deterministic artifact generation
for blockchain validation.

Public API Structure:
- core: Consensus-critical primitives (crypto, transforms, encoding, validation)
- protocol: API types and configuration (types, config, constants)
- inference: vLLM integration (model_runner, layer_hooks)
- runtime: FastAPI routes, async queue, HTTP callbacks
- utils: Environment variable management
"""

# Core consensus-critical functionality
from vllm.poc.core import (
    apply_haar_rotation,
    apply_householder,
    generate_householder_vector,
    # Geometric transforms
    generate_inputs,
    generate_target,
    murmur3_32,
    normal,
    normal_batch,
    random_pick_indices,
    # Crypto primitives
    seed_from_string,
    uniform,
    uniform_batch,
)

# Encoding and validation
from vllm.poc.core.encoding import decode_vector
from vllm.poc.core.validation import fraud_test, is_mismatch

# Inference hooks
from vllm.poc.inference import (
    LayerHouseholderHook,
    execute_poc_forward,
    poc_forward_context,
)

# Protocol types and configuration
from vllm.poc.protocol import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    # Constants
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
    Artifact,
    ArtifactBatchMeta,
    ArtifactValidationStats,
    Encoding,
    PoCConfig,
    # Types
    PoCParams,
    # Config
    PoCState,
)

# Runtime API
from vllm.poc.runtime import router as poc_router

__all__ = [
    # Core crypto
    "seed_from_string",
    "murmur3_32",
    "uniform",
    "normal",
    "uniform_batch",
    "normal_batch",
    # Core transforms
    "generate_inputs",
    "generate_target",
    "generate_householder_vector",
    "apply_householder",
    "apply_haar_rotation",
    "random_pick_indices",
    # Protocol types
    "PoCParams",
    "Artifact",
    "ArtifactBatchMeta",
    "ArtifactValidationStats",
    "Encoding",
    # Protocol config
    "PoCState",
    "PoCConfig",
    # Protocol constants
    "DEFAULT_K_DIM",
    "DEFAULT_DIST_THRESHOLD",
    "DEFAULT_P_MISMATCH",
    "DEFAULT_FRAUD_THRESHOLD",
    # Encoding
    "decode_vector",
    # Validation
    "fraud_test",
    "is_mismatch",
    # Inference
    "execute_poc_forward",
    "LayerHouseholderHook",
    "poc_forward_context",
    # Runtime
    "poc_router",
]
