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


def _ensure_1d_finite(vec: np.ndarray, *, name: str) -> np.ndarray:
    """Validate that vec is a finite 1D vector; return float64 view."""
    if vec.ndim != 1:
        raise ValueError(f"{name} must be 1D, got shape={vec.shape}")
    if not np.all(np.isfinite(vec)):
        raise ValueError(f"{name} contains non-finite values")
    return vec.astype(np.float64, copy=False)


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute L2 distance with float64 accumulation."""
    diff = a - b
    # dot(diff, diff) is explicit and stable in float64.
    return float(np.sqrt(float(np.dot(diff, diff))))


def is_mismatch(
    received_vector: np.ndarray,
    expected_b64: str,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> bool:
    """Check if vectors differ beyond threshold.

    Returns True on:
    - decode errors
    - non-finite values
    - shape mismatch
    - L2 distance > dist_threshold

    Args:
        received_vector: Locally computed vector (expected 1D).
        expected_b64: Base64-encoded expected vector.
        dist_threshold: L2 distance threshold for mismatch.
    """
    if dist_threshold < 0:
        raise ValueError(f"dist_threshold must be >= 0, got {dist_threshold}")

    try:
        expected_vector = decode_vector(expected_b64)
    except Exception:
        return True

    try:
        received = _ensure_1d_finite(received_vector, name="received_vector")
        expected = _ensure_1d_finite(expected_vector, name="expected_vector")
    except ValueError:
        return True

    if received.shape != expected.shape:
        return True
    distance = _l2_distance(received, expected)
    print(f"expected {expected_b64}, received {encode_vector(received)}, distance: {distance:.6f}")
    return distance > float(dist_threshold)


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
    if n_mismatch < 0:
        raise ValueError(f"n_mismatch must be >= 0, got {n_mismatch}")
    if n_total < 0:
        raise ValueError(f"n_total must be >= 0, got {n_total}")
    if n_mismatch > n_total:
        raise ValueError(f"n_mismatch cannot exceed n_total ({n_mismatch} > {n_total})")
    if not (0.0 <= p_mismatch <= 1.0):
        raise ValueError(f"p_mismatch must be in [0, 1], got {p_mismatch}")
    if not (0.0 <= fraud_threshold <= 1.0):
        raise ValueError(f"fraud_threshold must be in [0, 1], got {fraud_threshold}")

    if n_total == 0:
        return 1.0, False

    result = binomtest(
        k=n_mismatch, n=n_total, p=float(p_mismatch), alternative="greater"
    )
    p_value = float(result.pvalue)
    return p_value, (p_value < float(fraud_threshold))


def build_encoding(k_dim: int) -> Encoding:
    """Build Encoding object for given k_dim."""
    if k_dim <= 0:
        raise ValueError(f"k_dim must be > 0, got {k_dim}")
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
    """Validate computed artifacts against expected vectors; return stats."""
    n_total = 0
    n_mismatch = 0
    mismatch_nonces: list[int] = []

    for artifact in computed_artifacts:
        nonce = int(artifact.nonce)
        expected_b64 = expected_map.get(nonce)
        if expected_b64 is None:
            continue
        n_total += 1
        received_vector = decode_vector(artifact.vector_b64)
        if received_vector.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue
        if is_mismatch(received_vector, expected_b64, dist_threshold=dist_threshold):
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
