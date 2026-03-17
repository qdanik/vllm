"""PoC constants and defaults.

Central location for PoC defaults used across protocol, runtime, and v1.
"""

# Consensus validation parameters
DEFAULT_DIST_THRESHOLD: float = 0.4
DEFAULT_P_MISMATCH: float = 0.1
DEFAULT_FRAUD_THRESHOLD: float = 0.05

# Default k for PoC rotations / artifacts
DEFAULT_K_DIM: int = 12

# Runtime configuration (non-consensus)
POC_CHAT_BUSY_BACKOFF_SEC: float = 0.05
POC_CALLBACK_RETRY_BACKOFF_SEC: float = 1.0
POC_CALLBACK_RETRY_MAX_BACKOFF_SEC: float = 30.0

# Smaller priority values are scheduled first in PriorityRequestQueue.
# Chat defaults to 0, so PoC must be > 0 to yield under load.
POC_REQUEST_PRIORITY: int = 100

__all__ = [
    "DEFAULT_DIST_THRESHOLD",
    "DEFAULT_P_MISMATCH",
    "DEFAULT_FRAUD_THRESHOLD",
    "DEFAULT_K_DIM",
    "POC_CHAT_BUSY_BACKOFF_SEC",
    "POC_CALLBACK_RETRY_BACKOFF_SEC",
    "POC_CALLBACK_RETRY_MAX_BACKOFF_SEC",
    "POC_REQUEST_PRIORITY",
]
