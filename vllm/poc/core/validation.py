"""Consensus-critical artifact validation.

Fraud detection via binomial test (scipy.stats.binomtest with alternative='greater').
Statistical test parameters MUST NOT change without golden test updates.

DO NOT MODIFY without updating POC_CONSENSUS_INVARIANTS.md and golden tests.
"""


import numpy as np
from scipy.stats import binomtest

from vllm.poc.core.encoding import decode_vector

# Default constants - re-exported from protocol layer
DEFAULT_DIST_THRESHOLD = 0.4
DEFAULT_FRAUD_THRESHOLD = 0.05
DEFAULT_P_MISMATCH = 0.0


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
    distance = float(np.linalg.norm(computed_vector - received))
    return distance > dist_threshold


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> tuple[float, bool]:
    """
    Run binomial test for fraud detection.

    CONSENSUS-CRITICAL: Uses scipy.stats.binomtest with
    alternative='greater'. This tests whether n_mismatch is
    significantly higher than expected under p_mismatch.

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


def compare_artifacts(
    computed_vectors: list[np.ndarray],
    received_artifacts: list[any],  # List of objects with .nonce and .vector_b64
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> tuple[int, list[int]]:
    """
    Compare computed vectors against received artifacts.

    Args:
        computed_vectors: List of computed FP32 vectors
        received_artifacts: List of received Artifact objects (with .nonce, .vector_b64)
        dist_threshold: L2 distance threshold for mismatch

    Returns:
        (n_mismatch, mismatch_nonces)
    """
    n_mismatch = 0
    mismatch_nonces = []

    for vec, artifact in zip(computed_vectors, received_artifacts):
        if is_mismatch(vec, artifact.vector_b64, dist_threshold):
            n_mismatch += 1
            mismatch_nonces.append(artifact.nonce)

    return n_mismatch, mismatch_nonces
