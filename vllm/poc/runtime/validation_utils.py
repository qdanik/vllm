"""Helpers for PoC runtime validation and payloads."""

from __future__ import annotations

from typing import Any

from vllm.poc.core.encoding import decode_vector
from vllm.poc.core.validation import fraud_test, is_mismatch


def build_encoding(k_dim: int) -> dict[str, Any]:
    return {"dtype": "f16", "k_dim": k_dim, "endian": "le"}


def build_artifacts_payload(
    nonces: list[int],
    vectors_b64: list[str],
) -> list[dict[str, Any]]:
    return [
        {"nonce": nonce, "vector_b64": vector_b64}
        for nonce, vector_b64 in zip(nonces, vectors_b64)
    ]


def validate_artifacts(
    computed_artifacts: list[dict[str, Any]],
    expected_map: dict[int, str],
    *,
    dist_threshold: float,
    p_mismatch: float,
    fraud_threshold: float,
) -> dict[str, Any]:
    n_mismatch = 0
    mismatch_nonces: list[int] = []
    n_checked = 0

    for artifact in computed_artifacts:
        nonce = int(artifact["nonce"])
        expected_b64 = expected_map.get(nonce)
        if expected_b64 is None:
            continue
        n_checked += 1
        computed_vec = decode_vector(artifact["vector_b64"])
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

    return {
        "n_total": n_checked,
        "n_mismatch": n_mismatch,
        "mismatch_nonces": mismatch_nonces,
        "p_value": p_value,
        "fraud_detected": fraud_detected,
    }
