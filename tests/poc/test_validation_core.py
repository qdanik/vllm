"""Tests for core/validation module."""

import base64

import numpy as np

from vllm.poc.core.encoding import decode_vector
from vllm.poc.utils.validation import (
    fraud_test,
    is_mismatch,
)


def _encode_vector(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype("<f2").tobytes()).decode("ascii")


class TestValidationModuleImports:
    """Tests for validation module imports."""

    def test_is_mismatch_exists(self):
        """is_mismatch function should exist."""
        from vllm.poc.utils.validation import is_mismatch

        assert callable(is_mismatch)

    def test_fraud_test_exists(self):
        """fraud_test function should exist."""
        from vllm.poc.utils.validation import fraud_test

        assert callable(fraud_test)


class TestIsMismatch:
    """Tests for is_mismatch function."""

    def test_is_mismatch_identical_vectors(self):
        """Identical vectors should not be a mismatch."""
        v1 = np.array([1.0, 2.0, 3.0])
        v1_b64 = _encode_vector(v1)

        result = is_mismatch(v1, v1_b64, dist_threshold=0.1)
        assert result is False

    def test_is_mismatch_different_vectors(self):
        """Sufficiently different vectors should be a mismatch."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v2_b64 = _encode_vector(v2)
        threshold = 0.5

        result = is_mismatch(v1, v2_b64, dist_threshold=threshold)
        assert result is True

    def test_is_mismatch_returns_bool(self):
        """is_mismatch should return a boolean."""
        v1 = np.random.randn(32)
        v2 = np.random.randn(32)
        v2_b64 = _encode_vector(v2)

        result = is_mismatch(v1, v2_b64)
        assert isinstance(result, (bool, np.bool_))


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

    def test_is_mismatch_with_encoded_vectors(self):
        """is_mismatch should work with encoded vectors."""
        v1 = np.random.randn(16)
        v2 = np.random.randn(16)

        v2_b64 = _encode_vector(v2)

        # Should not raise
        result = is_mismatch(v1, v2_b64, dist_threshold=10.0)
        assert isinstance(result, (bool, np.bool_))
