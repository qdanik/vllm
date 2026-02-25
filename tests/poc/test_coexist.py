# ruff: noqa: E501
"""Tests for PoC+Chat coexistence through the direct RPC path."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from vllm.poc.server.compute import generation_loop
from vllm.poc.server.models import PoCConfig, PoCGenerationStats


class TestGenerationLoopBackoff:
    @pytest.mark.asyncio
    async def test_generation_loop_backs_off_on_timeout(self):
        engine_client = AsyncMock()
        stop_event = asyncio.Event()

        config = PoCConfig(
            block_hash="hash",
            block_height=100,
            public_key="key",
            node_id=0,
            node_count=1,
            group_id=0,
            n_groups=1,
            seq_len=256,
            k_dim=12,
        )
        stats = PoCGenerationStats()

        call_count = 0

        async def _mock_collective_rpc(method, timeout=None, args=(), kwargs=None):
            _ = method, timeout, kwargs
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise TimeoutError("engine busy")
            stop_event.set()
            nonces = list(args[2])
            return [{"nonces": nonces, "vectors_b64": ["AAAA" for _ in nonces]}]

        engine_client.collective_rpc = _mock_collective_rpc
        engine_client.vllm_config = MagicMock()
        engine_client.vllm_config.model_config = MagicMock()
        engine_client.vllm_config.model_config.get_hidden_size.return_value = 4096

        with patch("vllm.poc.server.compute.POC_CHAT_BUSY_BACKOFF_SEC", 0.001):
            task = asyncio.create_task(
                generation_loop(engine_client, stop_event, None, config, stats)
            )
            await asyncio.sleep(0.1)
            stop_event.set()
            with contextlib.suppress(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=1.0)

        assert call_count >= 2
