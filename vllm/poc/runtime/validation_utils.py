"""Helpers for PoC runtime validation and payloads."""

from __future__ import annotations

from vllm.poc.constants import DEFAULT_K_DIM
from vllm.poc.core.encoding import decode_vector
from vllm.poc.core.validation import fraud_test, is_mismatch
from vllm.poc.protocol.types import Artifact, ArtifactValidationStats, Encoding


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
    n_checked = 0

    for artifact in computed_artifacts:
        nonce = int(artifact.nonce)
        expected_b64 = expected_map.get(nonce)
        if expected_b64 is None:
            continue
        n_checked += 1
        computed_vec = decode_vector(artifact.vector_b64)
        # Check vector dimension matches k_dim
        if computed_vec.shape != (k_dim,):
            n_mismatch += 1
            mismatch_nonces.append(nonce)
            continue
        if is_mismatch(
            computed_vec,
            expected_b64,
            dist_threshold=dist_threshold,
        ):
            n_mismatch += 1
            mismatch_nonces.append(nonce)

    p_value, fraud_detected = fraud_test(
        n_mismatch,
        n_checked,
        p_mismatch=p_mismatch,
        fraud_threshold=fraud_threshold,
    )

    return ArtifactValidationStats(
        n_total=n_checked,
        n_mismatch=n_mismatch,
        mismatch_nonces=mismatch_nonces,
        p_value=p_value,
        fraud_detected=fraud_detected,
    )
