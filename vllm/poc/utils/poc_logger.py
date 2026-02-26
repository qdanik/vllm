"""PoC-specific logger utilities.

We keep the underlying logger name as the module name (via __name__), but
prepend messages with a stable prefix so PoC logs are easy to spot.
"""

from __future__ import annotations

import logging
from typing import Any, MutableMapping

from vllm.logger import init_logger

_POC_LOG_PREFIX = "[PoCV2] "


class PoCLoggerAdapter(logging.LoggerAdapter):
    def process(self, msg: Any, kwargs: MutableMapping[str, Any]):
        # Preserve %-formatting behavior by only touching the message template.
        if isinstance(msg, str):
            return f"{_POC_LOG_PREFIX}{msg}", kwargs
        return f"{_POC_LOG_PREFIX}{msg!s}", kwargs


def init_poc_logger(name: str) -> logging.LoggerAdapter:
    """Initialize a PoC logger for a given module name.

    Keep using `__name__` at call sites so log records retain the original
    module name, while messages gain a `[PoCV2]` prefix.
    """

    base = init_logger(name)
    return PoCLoggerAdapter(base, {})
