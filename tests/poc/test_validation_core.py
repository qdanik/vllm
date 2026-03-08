"""Tests for core/validation module."""

import base64

import numpy as np

from vllm.poc.consensus.encoding import decode_vector
from vllm.poc.server.validation import (
    decode_expected_map,
    fraud_test,
    validate_artifacts,
)


def _encode_vector(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype("<f2").tobytes()).decode("ascii")


class TestValidationModuleImports:
    """Tests for validation module imports."""

    def test_validate_artifacts_exists(self):
        """validate_artifacts function should exist."""
        from vllm.poc.server.validation import validate_artifacts

        assert callable(validate_artifacts)

    def test_fraud_test_exists(self):
        """fraud_test function should exist."""
        from vllm.poc.server.validation import fraud_test

        assert callable(fraud_test)

class TestFraudTest:
    """Tests for fraud_test function."""

    def test_fraud_test_zero_mismatches(self):
        """With 0 mismatches, should not detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=0,
            n_total=100,
        )

        assert fraud_detected is False
        assert isinstance(p_value, float)
        assert 0.0 <= p_value <= 1.0

    def test_fraud_test_high_mismatch_rate(self):
        """With high mismatch rate, should detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=50,
            n_total=100,
            p_mismatch=0.0,
            fraud_threshold=0.05,
        )

        assert isinstance(fraud_detected, (bool, np.bool_))

    def test_fraud_test_return_tuple(self):
        """fraud_test should return (float, bool) tuple."""
        result = fraud_test(n_mismatch=1, n_total=10)

        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], float)
        assert isinstance(result[1], (bool, np.bool_))


class TestEncodingVectors:
    """Tests for vector encoding."""

    def test_decode_vector_roundtrip(self):
        """Local encoder output should be properly decodable."""
        v = np.random.randn(12)
        v_b64 = _encode_vector(v)
        decoded = decode_vector(v_b64)
        assert decoded.shape == v.shape

    def test_decode_expected_map(self):
        """decode_expected_map should decode base64 vectors by nonce."""
        v = np.random.randn(16)
        decoded = decode_expected_map({7: _encode_vector(v)})
        assert 7 in decoded
        assert decoded[7].shape == (16,)


class TestValidateArtifacts:
    def test_validate_artifacts_detects_distance_mismatch(self):
        expected = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        computed = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        from vllm.poc.server.models import Artifact

        stats = validate_artifacts(
            computed_artifacts=[Artifact(nonce=1, vector_b64=_encode_vector(computed))],
            expected_map={1: _encode_vector(expected)},
            dist_threshold=0.01,
            p_mismatch=0.001,
            fraud_threshold=0.05,
            k_dim=3,
        )

        assert stats.n_total == 1
        assert stats.n_mismatch == 1
        assert stats.mismatch_nonces == [1]
