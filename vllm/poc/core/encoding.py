"""Consensus-critical vector encoding/decoding.

FP32 → FP16 (little-endian) → base64 encoding for compact artifact storage.

⚠️ CONSENSUS-CRITICAL ⚠️
- Encoding format MUST NOT change without updating all validators.
- Endianness MUST remain little-endian ('<f2').
- No implicit platform-dependent dtype behavior.

This module is intentionally explicit and strict.
"""

from __future__ import annotations

import base64
import numpy as np


# Explicit dtype constants (do not change)
_DTYPE_F16_LE = np.dtype("<f2")  # little-endian float16
_DTYPE_F32 = np.dtype(np.float32)


def _require_1d(vec: np.ndarray) -> None:
    if vec.ndim != 1:
        raise ValueError(f"vector must be 1D, got shape={vec.shape}")


def encode_vector(vector: np.ndarray) -> str:
    """Encode FP32 vector to base64 FP16 little-endian.

    Conversion pipeline:
        FP32 numpy array
            -> cast to little-endian float16 ('<f2')
            -> raw bytes
            -> base64 ASCII string

    Args:
        vector: numpy array (expected float32, 1D)

    Returns:
        Base64-encoded ASCII string
    """
    if not isinstance(vector, np.ndarray):
        raise TypeError("vector must be a numpy.ndarray")

    _require_1d(vector)

    # Ensure canonical float32 input before cast
    vec_f32 = vector.astype(_DTYPE_F32, copy=False)

    # Convert to little-endian float16 explicitly
    vec_f16_le = vec_f32.astype(_DTYPE_F16_LE, copy=False)

    return base64.b64encode(vec_f16_le.tobytes()).decode("ascii")


def decode_vector(b64: str) -> np.ndarray:
    """Decode base64 FP16 little-endian to FP32.

    Decoding pipeline:
        base64 ASCII
            -> raw bytes
            -> little-endian float16 ('<f2')
            -> cast to float32

    Args:
        b64: Base64-encoded ASCII string

    Returns:
        float32 numpy array (1D)
    """
    if not isinstance(b64, str):
        raise TypeError("b64 must be a base64-encoded string")

    data = base64.b64decode(b64)

    if len(data) % _DTYPE_F16_LE.itemsize != 0:
        raise ValueError("Invalid byte length for FP16 vector")

    vec_f16 = np.frombuffer(data, dtype=_DTYPE_F16_LE)

    # Cast explicitly to canonical float32
    return vec_f16.astype(_DTYPE_F32)
