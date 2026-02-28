"""Helpers for PoC validation and payloads."""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest

from vllm.poc.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.core.encoding import decode_vector, encode_vector
from vllm.poc.protocol.runtime_types import Artifact, ArtifactValidationStats, Encoding

def is_mismatch(
    computed_vector: np.ndarray,
    received_b64: str,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> bool:
    """Check if vectors differ beyond threshold.

    Args:
        computed_vector: Computed FP32 vector
        received_b64: Base64-encoded received vector
        dist_threshold: L2 distance threshold for mismatch

    Returns:
        True if distance > threshold
    """
    received = decode_vector(received_b64)
    if not np.all(np.isfinite(received)):
        return True
    distance = float(np.linalg.norm(computed_vector - received))
    print(f"received: {received_b64}, computed: {encode_vector(computed_vector)}, distance: {distance}")
    return distance > dist_threshold


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> tuple[float, bool]:
    """
    Run binomial test for fraud detection.

    Args:
        n_mismatch: Number of nonces where vectors differ beyond threshold
        n_total: Total nonces checked
        p_mismatch: Expected mismatch rate for honest nodes (baseline)
        fraud_threshold: p-value below which fraud is detected

    Returns:
        (p_value, fraud_detected)
    """
    if n_total == 0:
        return 1.0, False

    result = binomtest(k=n_mismatch, n=n_total, p=p_mismatch, alternative="greater")
    p_value = float(result.pvalue)
    fraud_detected = p_value < fraud_threshold
    return p_value, fraud_detected


def build_encoding(k_dim: int) -> Encoding:
    return Encoding(k_dim=k_dim)


def validate_artifacts(
    computed_artifacts: list[Artifact],
    expected_map: dict[int, str],
    *,
    dist_threshold: float,
    p_mismatch: float,
    fraud_threshold: float,
    k_dim: int = DEFAULT_K_DIM,
) -> ArtifactValidationStats:
    n_mismatch = 0
    mismatch_nonces: list[int] = []
    n_total = 0

    for artifact in computed_artifacts:
        nonce = int(artifact.nonce)
        expected_b64 = expected_map.get(nonce)
        if expected_b64 is None:
            continue
        n_total += 1
        computed_vec = decode_vector(artifact.vector_b64)
        if computed_vec.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue
        if is_mismatch(computed_vec, expected_b64, dist_threshold=dist_threshold):
            n_mismatch += 1
            mismatch_nonces.append(nonce)

    p_value, fraud_detected = fraud_test(
        n_mismatch,
        n_total,
        p_mismatch=p_mismatch,
        fraud_threshold=fraud_threshold,
    )

    return ArtifactValidationStats(
        n_total=n_total,
        n_mismatch=n_mismatch,
        mismatch_nonces=mismatch_nonces,
        p_value=p_value,
        fraud_detected=fraud_detected,
    )
