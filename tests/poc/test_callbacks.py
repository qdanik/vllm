"""Tests for runtime/callbacks module."""

import asyncio
from unittest.mock import MagicMock

import pytest

from vllm.poc.protocol.callbacks import CallbackQueue, CallbackSender
from vllm.poc.protocol.enums import CallbackPath


class TestCallbackSender:
    """Tests for CallbackSender class."""

    def test_callback_sender_initialization(self):
        """CallbackSender should initialize correctly."""
        stop_event = asyncio.Event()
        sender = CallbackSender(
            callback_url="http://example.com/callback",
            stop_event=stop_event,
        )

        assert sender.callback_url == "http://example.com/callback"
        assert sender.stop_event is stop_event

    def test_callback_sender_has_methods(self):
        """CallbackSender should have expected methods."""
        stop_event = asyncio.Event()
        sender = CallbackSender(
            callback_url="http://example.com/callback",
            stop_event=stop_event,
        )

        # Check required methods
        assert hasattr(sender, "add_artifacts")
        assert hasattr(sender, "clear")
        assert hasattr(sender, "run")


class TestCallbackQueue:
    """Tests for CallbackQueue class."""

    def test_callback_queue_initialization(self):
        """CallbackQueue should initialize correctly."""
        stop_event = asyncio.Event()
        queue = CallbackQueue(stop_event=stop_event)

        assert queue.stop_event is stop_event
        assert hasattr(queue, "_queue")

    def test_callback_queue_has_enqueue(self):
        """CallbackQueue should have enqueue method."""
        stop_event = asyncio.Event()
        queue = CallbackQueue(stop_event=stop_event)

        assert callable(queue.enqueue)
        assert hasattr(queue, "pending_count")


class TestCallbackIntegration:
    """Integration tests for callback components."""

    def test_callback_path_enum(self):
        """CallbackPath enum should exist and have values."""
        assert CallbackPath is not None
        # Check that it's an enum
        members = list(CallbackPath.__members__.keys())
        assert len(members) > 0

    def test_callback_sender_buffering(self):
        """CallbackSender should have buffering capabilities."""
        stop_event = asyncio.Event()
        sender = CallbackSender(
            callback_url="http://example.com",
            stop_event=stop_event,
        )
        
        # Should have buffer tracking
        assert sender.buffered_count == 0
        

class TestCallbackQueueAsync:
    """Async tests for CallbackQueue."""

    @pytest.mark.asyncio
    async def test_callback_queue_has_start_stop(self):
        """CallbackQueue should have start and stop async methods."""
        stop_event = asyncio.Event()
        queue = CallbackQueue(stop_event=stop_event)
        
        assert hasattr(queue, "start")
        assert hasattr(queue, "stop")
        assert callable(queue.start)
        assert callable(queue.stop)
