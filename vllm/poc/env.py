"""PoC environment variables.

This module centralizes PoC-related env var parsing.
Matches the lazy __getattr__ pattern used in vllm/envs.py.
"""

from __future__ import annotations

import functools
import os
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    # Batch sizing / RPC
    POC_RPC_TIMEOUT_MS: int
    POC_BATCH_SIZE_DEFAULT: int
    POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT: bool

    # Callback sender
    POC_CALLBACK_INTERVAL_SEC: float
    POC_CALLBACK_MAX_ARTIFACTS: int
    POC_CALLBACK_MAX_RETRIES: int
    POC_CALLBACK_MAX_CONCURRENT: int
    POC_CALLBACK_QUEUE_SIZE: int
    POC_CALLBACK_RETRY_BACKOFF_SEC: float
    POC_CALLBACK_RETRY_MAX_BACKOFF_SEC: float
    POC_LOG_ARTIFACTS_JSON: bool

    # /generate queue
    POC_GENERATE_CHUNK_TIMEOUT_SEC: float
    POC_GENERATE_RESULT_TTL_SEC: float
    POC_MAX_QUEUED_NONCES: int

    # Scheduler / token budget
    POC_MAX_NUM_BATCHED_TOKENS: int
    POC_MAX_NUM_SEQS: int

    # Profiling
    POC_PROFILE_DIST_THRESHOLD: float
    POC_PROFILE_P_MISMATCH: float
    POC_PROFILE_FRAUD_THRESHOLD: float


environment_variables: dict[str, Callable[[], Any]] = {
    # Batch sizing / RPC
    "POC_RPC_TIMEOUT_MS": lambda: int(os.getenv("POC_RPC_TIMEOUT_MS", "60000")),
    "POC_BATCH_SIZE_DEFAULT": lambda: int(os.getenv("POC_BATCH_SIZE_DEFAULT", "32")),
    "POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT": lambda: os.getenv(
        "POC_FORCE_BATCH_SIZE_DEFAULT_ON_INIT", "1"
    )
    == "1",
    # Callback sender
    "POC_CALLBACK_INTERVAL_SEC": lambda: float(
        os.getenv("POC_CALLBACK_INTERVAL_SEC", "5")
    ),
    "POC_CALLBACK_MAX_ARTIFACTS": lambda: int(
        os.getenv("POC_CALLBACK_MAX_ARTIFACTS", "1000000")
    ),
    "POC_CALLBACK_MAX_RETRIES": lambda: int(
        os.getenv("POC_CALLBACK_MAX_RETRIES", "10")
    ),
    "POC_CALLBACK_MAX_CONCURRENT": lambda: int(
        os.getenv("POC_CALLBACK_MAX_CONCURRENT", "10")
    ),
    "POC_CALLBACK_QUEUE_SIZE": lambda: int(
        os.getenv("POC_CALLBACK_QUEUE_SIZE", "10000")
    ),
    "POC_CALLBACK_RETRY_BACKOFF_SEC": lambda: float(
        os.getenv("POC_CALLBACK_RETRY_BACKOFF_SEC", "1.0")
    ),
    "POC_CALLBACK_RETRY_MAX_BACKOFF_SEC": lambda: float(
        os.getenv("POC_CALLBACK_RETRY_MAX_BACKOFF_SEC", "30.0")
    ),
    "POC_LOG_ARTIFACTS_JSON": lambda: os.getenv("POC_LOG_ARTIFACTS_JSON", "0") == "1",
    # /generate queue
    "POC_GENERATE_CHUNK_TIMEOUT_SEC": lambda: float(
        os.getenv("POC_GENERATE_CHUNK_TIMEOUT_SEC", "60")
    ),
    "POC_GENERATE_RESULT_TTL_SEC": lambda: float(
        os.getenv("POC_GENERATE_RESULT_TTL_SEC", "300")
    ),
    "POC_MAX_QUEUED_NONCES": lambda: int(os.getenv("POC_MAX_QUEUED_NONCES", "100000")),
    # Scheduler / token budget
    "POC_MAX_NUM_BATCHED_TOKENS": lambda: int(
        os.getenv("POC_MAX_NUM_BATCHED_TOKENS", "0")
    ),
    "POC_MAX_NUM_SEQS": lambda: int(os.getenv("POC_MAX_NUM_SEQS", "256")),
    # profile_poc.py helpers
    "POC_PROFILE_DIST_THRESHOLD": lambda: float(
        os.getenv("POC_PROFILE_DIST_THRESHOLD", "0.4")
    ),
    "POC_PROFILE_P_MISMATCH": lambda: float(os.getenv("POC_PROFILE_P_MISMATCH", "0.1")),
    "POC_PROFILE_FRAUD_THRESHOLD": lambda: float(
        os.getenv("POC_PROFILE_FRAUD_THRESHOLD", "0.05")
    ),
}


def __getattr__(name: str):
    """Lazily evaluate PoC env vars.

    Matches the pattern used in `vllm/envs.py`.
    """

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
    return sorted(list(environment_variables.keys()))


def is_set(name: str) -> bool:
    """Check if an env variable is explicitly set."""

    if name in environment_variables:
        return name in os.environ
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
