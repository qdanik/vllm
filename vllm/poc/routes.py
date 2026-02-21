"""Deprecated: Use vllm.poc.runtime.routes instead.

This module exists for backward compatibility only.
"""
import warnings

warnings.warn(
    "vllm.poc.routes is deprecated, use vllm.poc.runtime.routes instead",
    DeprecationWarning,
    stacklevel=2,
)

from .runtime.routes import (
    router,
    calculate_optimal_batch_size,
    run_poc_request,
)

__all__ = [
    "router",
    "calculate_optimal_batch_size",
    "run_poc_request",
]
