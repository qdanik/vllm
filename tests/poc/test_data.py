"""Tests for PoC data types and helpers (artifact-based protocol)."""

# ruff: noqa: E501

import base64

import numpy as np

from vllm.poc.consensus.encoding import decode_vector
from vllm.poc.server.models import (
    Artifact,
    ArtifactValidationStats,
    Encoding,
    PoCConfig,
    PoCModelParams,
    PoCState,
)
from vllm.poc.server.schemas import ArtifactBatchSchema
from vllm.poc.server.validation import fraud_test


def _encode_vector(vector: np.ndarray) -> str:
    return base64.b64encode(vector.astype("<f2").tobytes()).decode("ascii")


class TestPoCConfig:
    def test_defaults(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
        )
        assert config.seq_len == 256
        assert config.k_dim == 12
        assert config.node_id == 0
        assert config.node_count == 1
        assert config.callback_url is None

    def test_all_fields(self):
        config = PoCConfig(
            block_hash="hash1",
            block_height=100,
            public_key="node1",
            node_id=2,
            node_count=4,
            seq_len=128,
            k_dim=8,
            callback_url="http://localhost:8080/callback",
        )
        assert config.node_id == 2
        assert config.node_count == 4
        assert config.k_dim == 8
        assert config.callback_url == "http://localhost:8080/callback"


class TestPoCState:
    def test_states_exist(self):
        assert PoCState.IDLE.value == "IDLE"
        assert PoCState.GENERATING.value == "GENERATING"
        assert PoCState.STOPPED.value == "STOPPED"


class TestPoCModelParams:
    def test_creation(self):
        params = PoCModelParams(model="Qwen/Qwen3-0.6B", seq_len=256, k_dim=12)
        assert params.model == "Qwen/Qwen3-0.6B"
        assert params.seq_len == 256
        assert params.k_dim == 12

    def test_default_k_dim(self):
        params = PoCModelParams(model="test-model", seq_len=128)
        assert params.k_dim == 12


class TestArtifact:
    def test_creation(self):
        artifact = Artifact(nonce=42, vector_b64="AAAAAAAAAAAAAAAA")
        assert artifact.nonce == 42
        assert artifact.vector_b64 == "AAAAAAAAAAAAAAAA"


class TestEncoding:
    def test_defaults(self):
        encoding = Encoding()
        assert encoding.dtype == "f16"
        assert encoding.k_dim == 12
        assert encoding.endian == "le"

    def test_custom(self):
        encoding = Encoding(dtype="f32", k_dim=8, endian="be")
        assert encoding.dtype == "f32"
        assert encoding.k_dim == 8
        assert encoding.endian == "be"


class TestVectorEncoding:
    def test_encode_decode_roundtrip(self):
        """Encode FP32 vector to base64, decode back to FP32."""
        original = np.array([0.1, 0.2, 0.3, -0.5, 1.0], dtype=np.float32)
        encoded = _encode_vector(original)
        decoded = decode_vector(encoded)

        # Should be very close (FP16 may lose some precision)
        np.testing.assert_allclose(decoded, original, rtol=1e-3)

    def test_encode_produces_base64(self):
        """Encoded string should be valid base64."""
        import base64

        vec = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = _encode_vector(vec)

        # Should be decodable as base64
        decoded_bytes = base64.b64decode(encoded)
        assert len(decoded_bytes) == 3 * 2  # 3 floats * 2 bytes each (FP16)

    def test_decode_correct_length(self):
        """Decoded vector should have correct length."""
        # Encode a 12-dim vector
        vec = np.random.randn(12).astype(np.float32)
        encoded = _encode_vector(vec)
        decoded = decode_vector(encoded)

        assert decoded.shape == (12,)

    def test_little_endian_format(self):
        """Verify little-endian byte order."""

        # Known value
        vec = np.array([1.0], dtype=np.float32)
        encoded = _encode_vector(vec)
        decoded_bytes = np.frombuffer(
            __import__("base64").b64decode(encoded), dtype="<f2"
        )

        # 1.0 in FP16 little-endian
        assert decoded_bytes[0] == np.float16(1.0)


class TestFraudTest:
    def test_no_mismatch_no_fraud(self):
        """Zero mismatches should not detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=0, n_total=100, p_mismatch=0.001, fraud_threshold=0.01
        )

        assert fraud_detected is False
        assert p_value > 0.01

    def test_high_mismatch_detects_fraud(self):
        """Many mismatches should detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=50,
            n_total=100,  # 50% mismatch rate
            p_mismatch=0.001,
            fraud_threshold=0.01,
        )

        assert fraud_detected is True
        assert p_value < 0.01

    def test_empty_no_fraud(self):
        """Empty batch should not detect fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=0, n_total=0, p_mismatch=0.001, fraud_threshold=0.01
        )

        assert fraud_detected is False
        assert p_value == 1.0

    def test_single_mismatch_small_sample(self):
        """Single mismatch in small sample may or may not be fraud."""
        p_value, fraud_detected = fraud_test(
            n_mismatch=1, n_total=10, p_mismatch=0.001, fraud_threshold=0.01
        )

        # With p_mismatch=0.001 and 1/10 mismatches, p_value should be low
        assert p_value < 0.1


class TestArtifactBatch:
    def test_creation(self):
        batch = ArtifactBatchSchema(
            public_key="node1",
            block_hash="hash1",
            block_height=100,
            node_id=0,
            artifacts=[Artifact(nonce=0, vector_b64="abc")],
            encoding=Encoding(k_dim=12),
        )

        assert batch.public_key == "node1"
        assert len(batch.artifacts) == 1


class TestArtifactValidationStats:
    def test_creation(self):
        result = ArtifactValidationStats(
            n_total=3,
            n_mismatch=1,
            mismatch_nonces=[2],
            p_value=0.01,
            fraud_detected=True,
        )

        assert result.n_total == 3
        assert result.n_mismatch == 1
        assert result.mismatch_nonces == [2]
