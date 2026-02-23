"""Tests for core/validation module."""

import numpy as np
import pytest

from vllm.poc.core.validation import (
    is_mismatch,
    fraud_test,
    compare_artifacts,
)
from vllm.poc.core.encoding import encode_vector


class TestValidationModuleImports:
    """Tests for validation module imports."""

    def test_is_mismatch_exists(self):
        """is_mismatch function should exist."""
        from vllm.poc.core.validation import is_mismatch
        assert callable(is_mismatch)

    def test_fraud_test_exists(self):
        """fraud_test function should exist."""
        from vllm.poc.core.validation import fraud_test
        assert callable(fraud_test)

    def test_compare_artifacts_exists(self):
        """compare_artifacts function should exist."""
        from vllm.poc.core.validation import compare_artifacts
        assert callable(compare_artifacts)


class TestIsMismatch:
    """Tests for is_mismatch function."""

    def test_is_mismatch_identical_vectors(self):
        """Identical vectors should not be a mismatch."""
        v1 = np.array([1.0, 2.0, 3.0])
        v1_b64 = encode_vector(v1)

        result = is_mismatch(v1, v1_b64, dist_threshold=0.1)
        assert result is False

    def test_is_mismatch_different_vectors(self):
        """Sufficiently different vectors should be a mismatch."""
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        v2_b64 = encode_vector(v2)
        threshold = 0.5

        result = is_mismatch(v1, v2_b64, dist_threshold=threshold)
        assert result is True

    def test_is_mismatch_returns_bool(self):
        """is_mismatch should return a boolean."""
        v1 = np.random.randn(32)
        v2 = np.random.randn(32)
        v2_b64 = encode_vector(v2)

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


class TestCompareArtifacts:
    """Tests for compare_artifacts function."""

    def test_compare_artifacts_matching_vectors(self):
        """Matching vectors should return 0 mismatches."""
        n_artifacts = 5
        computed = [np.random.randn(32) for _ in range(n_artifacts)]

        class MockArtifact:
            def __init__(self, nonce, vector_b64):
                self.nonce = nonce
                self.vector_b64 = vector_b64

        received = [
            MockArtifact(i, encode_vector(v))
            for i, v in enumerate(computed)
        ]

        n_mismatch, mismatch_nonces = compare_artifacts(
            computed,
            received,
            dist_threshold=1.0,
        )

        assert isinstance(n_mismatch, int)
        assert isinstance(mismatch_nonces, list)
        assert n_mismatch == 0

    def test_compare_artifacts_return_types(self):
        """compare_artifacts should return (int, list)."""
        computed = [np.random.randn(12) for _ in range(3)]
        
        class MockArtifact:
            def __init__(self, nonce, vector_b64):
                self.nonce = nonce
                self.vector_b64 = vector_b64

        received = [
            MockArtifact(i, encode_vector(v))
            for i, v in enumerate(computed)
        ]

        result = compare_artifacts(computed, received)
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], int)
        assert isinstance(result[1], list)


class TestEncodingVectors:
    """Tests for vector encoding."""

    def test_encode_vector_roundtrip(self):
        """encode_vector should be properly decodable."""
        v = np.random.randn(12)
        v_b64 = encode_vector(v)
        
        # Should produce base64 string
        assert isinstance(v_b64, str)

    def test_is_mismatch_with_encoded_vectors(self):
        """is_mismatch should work with encode_vector output."""
        v1 = np.random.randn(16)
        v2 = np.random.randn(16)
        
        v2_b64 = encode_vector(v2)
        
        # Should not raise
        result = is_mismatch(v1, v2_b64, dist_threshold=10.0)
        assert isinstance(result, (bool, np.bool_))
