"""PoC environment variables.

This module centralizes PoC-related env var parsing.
Matches the lazy __getattr__ pattern used in vllm/envs.py.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from vllm.poc.protocol.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_P_MISMATCH,
)

if TYPE_CHECKING:
    # Batch sizing / RPC
    POC_RPC_TIMEOUT_MS: int
    POC_BATCH_SIZE_DEFAULT: int

    # Callback sender
    POC_CALLBACK_INTERVAL_SEC: float
    POC_CALLBACK_MAX_ARTIFACTS: int
    POC_CALLBACK_MAX_RETRIES: int
    POC_CALLBACK_MAX_CONCURRENT: int
    POC_CALLBACK_QUEUE_SIZE: int
    POC_LOG_ARTIFACTS_JSON: bool

    # /generate queue
    POC_GENERATE_CHUNK_TIMEOUT_SEC: float
    POC_GENERATE_RESULT_TTL_SEC: float
    POC_MAX_QUEUED_NONCES: int

    # Profiling
    POC_PROFILE_DIST_THRESHOLD: float
    POC_PROFILE_P_MISMATCH: float
    POC_PROFILE_FRAUD_THRESHOLD: float


environment_variables: dict[str, Callable[[], Any]] = {
    # Batch sizing / RPC
    "POC_RPC_TIMEOUT_MS": lambda: int(os.getenv("POC_RPC_TIMEOUT_MS", "60000")),
    "POC_BATCH_SIZE_DEFAULT": lambda: int(os.getenv("POC_BATCH_SIZE_DEFAULT", "32")),
    # Callback sender
    "POC_CALLBACK_INTERVAL_SEC": lambda: float(os.getenv("POC_CALLBACK_INTERVAL_SEC", "5")),
    "POC_CALLBACK_MAX_ARTIFACTS": lambda: int(os.getenv("POC_CALLBACK_MAX_ARTIFACTS", "1000000")),
    "POC_CALLBACK_MAX_RETRIES": lambda: int(os.getenv("POC_CALLBACK_MAX_RETRIES", "10")),
    "POC_CALLBACK_MAX_CONCURRENT": lambda: int(os.getenv("POC_CALLBACK_MAX_CONCURRENT", "10")),
    "POC_CALLBACK_QUEUE_SIZE": lambda: int(os.getenv("POC_CALLBACK_QUEUE_SIZE", "10000")),
    "POC_LOG_ARTIFACTS_JSON": lambda: os.getenv("POC_LOG_ARTIFACTS_JSON", "0") == "1",
    # /generate queue
    "POC_GENERATE_CHUNK_TIMEOUT_SEC": lambda: float(
        os.getenv("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60")
    ),
    "POC_GENERATE_RESULT_TTL_SEC": lambda: float(os.getenv("POC_GENERATE_RESULT_TTL_SEC", "300")),
    "POC_MAX_QUEUED_NONCES": lambda: int(os.getenv("POC_MAX_QUEUED_NONCES", "100000")),
    # profile_poc.py helpers
    "POC_PROFILE_DIST_THRESHOLD": lambda: float(
        os.getenv("POC_PROFILE_DIST_THRESHOLD", str(DEFAULT_DIST_THRESHOLD))
    ),
    "POC_PROFILE_P_MISMATCH": lambda: float(
        os.getenv("POC_PROFILE_P_MISMATCH", str(DEFAULT_P_MISMATCH))
    ),
    "POC_PROFILE_FRAUD_THRESHOLD": lambda: float(
        os.getenv("POC_PROFILE_FRAUD_THRESHOLD", str(DEFAULT_FRAUD_THRESHOLD))
    ),
}


def __getattr__(name: str):
    """Lazily evaluate PoC env vars.

    Matches the pattern used in `vllm/envs.py`.
    """
    # Handle constants from protocol.constants
    if name in (
        "DEFAULT_DIST_THRESHOLD",
        "DEFAULT_P_MISMATCH",
        "DEFAULT_FRAUD_THRESHOLD",
        "DEFAULT_K_DIM",
        "POC_CHAT_BUSY_BACKOFF_SEC",
        "POC_CALLBACK_RETRY_BACKOFF_SEC",
        "POC_CALLBACK_RETRY_MAX_BACKOFF_SEC",
    ):
        import vllm.poc.protocol.constants as constants

        return getattr(constants, name)

    if name in environment_variables:
        return environment_variables[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _is_envs_cache_enabled() -> bool:
    global __getattr__
    return hasattr(__getattr__, "cache_clear")


def enable_envs_cache() -> None:
    """Cache env var values after initialization for performance."""
    if _is_envs_cache_enabled():
        return
    global __getattr__
    __getattr__ = functools.cache(__getattr__)
    for key in environment_variables:
        __getattr__(key)


def disable_envs_cache() -> None:
    """Disable cached env var values (useful for tests)."""
    global __getattr__
    if _is_envs_cache_enabled():
        __getattr__ = __getattr__.__wrapped__


def __dir__():
    return sorted(
        list(environment_variables.keys())
        + [
            "DEFAULT_DIST_THRESHOLD",
            "DEFAULT_P_MISMATCH",
            "DEFAULT_FRAUD_THRESHOLD",
            "DEFAULT_K_DIM",
            "POC_CHAT_BUSY_BACKOFF_SEC",
            "POC_CALLBACK_RETRY_BACKOFF_SEC",
            "POC_CALLBACK_RETRY_MAX_BACKOFF_SEC",
        ]
    )


def is_set(name: str) -> bool:
    """Check if an env variable is explicitly set."""
    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
