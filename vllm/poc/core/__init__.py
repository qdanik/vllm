"""Consensus-critical core primitives for PoC.

WARNING: This module contains consensus-critical code.
Any changes MUST preserve bit-exact determinism.
"""

from vllm.poc.core.crypto import (
    murmur3_32,
    murmur3_32_batch,
    normal,
    normal_batch,
    seed_from_string,
    uniform,
    uniform_batch,
)
from vllm.poc.core.encoding import (
    decode_vector,
    encode_vector,
)
from vllm.poc.core.transforms import (
    apply_haar_rotation,
    apply_householder,
    generate_householder_vector,
    generate_inputs,
    generate_target,
    random_pick_indices,
)
from vllm.poc.core.validation import (
    compare_artifacts,
    fraud_test,
    is_mismatch,
)

__all__ = [
    # Crypto/RNG
    "seed_from_string",
    "murmur3_32",
    "uniform",
    "normal",
    "murmur3_32_batch",
    "uniform_batch",
    "normal_batch",
    # Transforms
    "generate_inputs",
    "generate_target",
    "generate_householder_vector",
    "apply_householder",
    "apply_haar_rotation",
    "random_pick_indices",
    # Encoding
    "encode_vector",
    "decode_vector",
    # Validation
    "fraud_test",
    "is_mismatch",
    "compare_artifacts",
]
