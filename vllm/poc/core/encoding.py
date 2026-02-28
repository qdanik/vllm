"""Consensus-critical vector encoding/decoding.

FP32 → FP16 little-endian → base64 encoding for compact artifact storage.
Encoding format MUST NOT change without updating all validators.
"""

import base64

import numpy as np


def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian.

    CONSENSUS-CRITICAL: Use '<f2' (little-endian float16) format.

    Args:
        vector: FP32 numpy array

    Returns:
        Base64-encoded string
    """
    f16 = vector.astype("<f2")  # '<f2' = little-endian float16
    return base64.b64encode(f16.tobytes()).decode("ascii")


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32.

    CONSENSUS-CRITICAL: Use '<f2' (little-endian float16) format.

    Args:
        b64: Base64-encoded string

    Returns:
        FP32 numpy array
    """
    data = base64.b64decode(b64)
    f16 = np.frombuffer(data, dtype="<f2")
    return f16.astype(np.float32)
