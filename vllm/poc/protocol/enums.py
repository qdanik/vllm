"""Protocol enums for PoC runtime.

These enums keep status strings consistent across API responses, queue records,
and callbacks.
"""

from __future__ import annotations

from enum import Enum


class ApiStatus(str, Enum):
    OK = "OK"


class GenerateStatus(str, Enum):
    QUEUED = "queued"
    COMPLETED = "completed"


class GenerateResultStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CallbackPath(str, Enum):
    GENERATED = "generated"
    VALIDATED = "validated"
