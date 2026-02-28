"""Helpers for PoC validation and payloads (v2: safe expected cache).

Notes:
- Add optional pre-decoding of expected_map base64 vectors into float32 numpy
  arrays to avoid repeated base64 decode in validation loops.
- distance metric (L2)
- mismatch semantics
- fraud_test computation

This is safe for determinism: it is a pure performance optimization.
"""

from __future__ import annotations

from collections.abc import Mapping

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


def build_encoding(k_dim: int) -> Encoding:
    return Encoding(k_dim=int(k_dim))


def is_mismatch(
    computed_vector: np.ndarray,
    received_b64: str,
    dist_threshold: float = DEFAULT_DIST_THRESHOLD,
) -> bool:
    """Check if vectors differ beyond threshold."""
    received = decode_vector(received_b64)
    if not np.all(np.isfinite(received)):
        return True
    distance = float(np.linalg.norm(computed_vector - received))
    print(
        f"expected: {received_b64}, "
        f"computed: {encode_vector(computed_vector)}, "
        f"distance: {distance:.8f}"
    )
    return distance > float(dist_threshold)


def fraud_test(
    n_mismatch: int,
    n_total: int,
    p_mismatch: float = DEFAULT_P_MISMATCH,
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD,
) -> tuple[float, bool]:
    """Run binomial test for fraud detection."""
    if int(n_total) == 0:
        return 1.0, False

    res = binomtest(
        k=int(n_mismatch),
        n=int(n_total),
        p=float(p_mismatch),
        alternative="greater",
    )
    p_value = float(res.pvalue)
    return p_value, p_value < float(fraud_threshold)


# ---------------------------------------------------------------------------
# v2: expected vector cache helpers
# ---------------------------------------------------------------------------


def decode_expected_map(expected_map: Mapping[int, str]) -> dict[int, np.ndarray]:
    """Decode expected_map[nonce] base64 -> float32 numpy array.

    SAFE: purely avoids repeated base64 decode work.
    """
    out: dict[int, np.ndarray] = {}
    for nonce, b64 in expected_map.items():
        out[int(nonce)] = decode_vector(b64)
    return out


def validate_artifacts(
    computed_artifacts: list[Artifact],
    expected_map: Mapping[int, str] | None = None,
    *,
    expected_vec_map: Mapping[int, np.ndarray] | None = None,
    dist_threshold: float,
    p_mismatch: float,
    fraud_threshold: float,
    k_dim: int = DEFAULT_K_DIM,
) -> ArtifactValidationStats:
    """Validate computed artifacts against expected vectors.

    Provide either:
    - expected_map (nonce -> base64 vector), or
    - expected_vec_map (nonce -> decoded float32 vector)

    If both are provided, expected_vec_map takes precedence.
    """
    if expected_vec_map is None and expected_map is None:
        raise ValueError("Either expected_map or expected_vec_map must be provided")

    k_dim = int(k_dim)

    n_total = 0
    n_mismatch = 0
    mismatch_nonces: list[int] = []

    # Fast path: decoded expected vectors.
    if expected_vec_map is not None:
        for artifact in computed_artifacts:
            nonce = int(artifact.nonce)
            expected_vec = expected_vec_map.get(nonce)
            if expected_vec is None:
                continue

            n_total += 1
            computed_vec = decode_vector(artifact.vector_b64)

            if computed_vec.shape != (k_dim,) or expected_vec.shape != (k_dim,):
                n_mismatch += 1
                mismatch_nonces.append(nonce)
                continue

            if not np.all(np.isfinite(computed_vec)):
                n_mismatch += 1
                mismatch_nonces.append(nonce)
                continue

            distance = float(np.linalg.norm(computed_vec - expected_vec))
            if distance > float(dist_threshold):
                n_mismatch += 1
                mismatch_nonces.append(nonce)

    else:
        # Legacy path: base64 expected vectors.
        assert expected_map is not None
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

            if is_mismatch(
                computed_vec,
                expected_b64,
                dist_threshold=float(dist_threshold),
            ):
                n_mismatch += 1
                mismatch_nonces.append(nonce)

    p_value, fraud_detected = fraud_test(
        n_mismatch,
        n_total,
        p_mismatch=float(p_mismatch),
        fraud_threshold=float(fraud_threshold),
    )

    return ArtifactValidationStats(
        n_total=n_total,
        n_mismatch=n_mismatch,
        mismatch_nonces=mismatch_nonces,
        p_value=p_value,
        fraud_detected=fraud_detected,
    )
