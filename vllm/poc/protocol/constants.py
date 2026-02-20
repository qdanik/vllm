"""PoC protocol constants and default values.

Central location for consensus-critical constants and default parameters.
"""

# Consensus validation parameters
DEFAULT_DIST_THRESHOLD: float = 0.4  # L2 distance threshold for vector mismatch
DEFAULT_P_MISMATCH: float = 0.1  # Expected mismatch rate for honest nodes
DEFAULT_FRAUD_THRESHOLD: float = 0.05  # p-value threshold for fraud detection

# Default k for PoC rotations / artifacts
DEFAULT_K_DIM: int = 12

# Runtime configuration (non-consensus)
POC_CHAT_BUSY_BACKOFF_SEC: float = 0.05  # Cooperative backoff when chat is busy
POC_CALLBACK_RETRY_BACKOFF_SEC: float = 1.0  # Initial callback retry backoff
POC_CALLBACK_RETRY_MAX_BACKOFF_SEC: float = 30.0  # Max callback retry backoff

__all__ = [
    "DEFAULT_DIST_THRESHOLD",
    "DEFAULT_P_MISMATCH",
    "DEFAULT_FRAUD_THRESHOLD",
    "DEFAULT_K_DIM",
    "POC_CHAT_BUSY_BACKOFF_SEC",
    "POC_CALLBACK_RETRY_BACKOFF_SEC",
    "POC_CALLBACK_RETRY_MAX_BACKOFF_SEC",
]
