"""Tests for V1 integration components (gpu_runner_poc, async_engine_poc)."""

import pytest

from vllm.poc.v1.params import PoCParams
from vllm.poc.v1.constants import POC_REQUEST_PRIORITY


class TestGpuRunnerPocImports:
    """Tests for gpu_runner_poc module imports."""

    def test_batch_has_poc_exists(self):
        """batch_has_poc function should exist."""
        from vllm.poc.v1.gpu_runner import batch_has_poc
        assert callable(batch_has_poc)

    def test_fill_poc_inputs_embeds_exists(self):
        """fill_poc_inputs_embeds function should exist."""
        from vllm.poc.v1.gpu_runner import fill_poc_inputs_embeds
        assert callable(fill_poc_inputs_embeds)

    def test_extract_poc_results_exists(self):
        """extract_poc_results function should exist."""
        from vllm.poc.v1.gpu_runner import extract_poc_results
        assert callable(extract_poc_results)


class TestAsyncEnginePocImports:
    """Tests for async_engine_poc module imports."""

    def test_poc_compute_impl_exists(self):
        """poc_compute_impl function should exist."""
        from vllm.poc.v1.async_engine import poc_compute_impl
        assert callable(poc_compute_impl)

    def test_poc_compute_impl_is_async(self):
        """poc_compute_impl should be async function."""
        import inspect
        from vllm.poc.v1.async_engine import poc_compute_impl
        assert inspect.iscoroutinefunction(poc_compute_impl)


class TestPoCParams:
    """Tests for PoCParams data class."""

    def test_poc_params_initialization(self):
        """PoCParams should initialize with required fields."""
        from vllm.poc.v1.params import PoCParams
        params = PoCParams(
            block_hash="hash123",
            public_key="pubkey456",
            block_height=789,
            nonce=42,
            seq_len=32,
            k_dim=12,
        )
        
        assert params.block_hash == "hash123"
        assert params.public_key == "pubkey456"
        assert params.block_height == 789
        assert params.nonce == 42
        assert params.seq_len == 32
        assert params.k_dim == 12

    def test_poc_params_default_values(self):
        """PoCParams should have sensible defaults."""
        from vllm.poc.v1.params import PoCParams
        params = PoCParams(
            block_hash="hash",
            public_key="key",
            block_height=100,
            nonce=1,
            seq_len=32,
        )
        
        # Should have default k_dim
        assert params.k_dim == 12  # or whatever the default is
        # Should have default optional fields
        assert hasattr(params, 'r_target')
        assert hasattr(params, 'return_vectors')


class TestV1IntegrationConstants:
    """Tests for v1_integration constants."""

    def test_poc_request_priority_value(self):
        """POC_REQUEST_PRIORITY should be higher than chat (0)."""
        assert POC_REQUEST_PRIORITY > 0
        assert POC_REQUEST_PRIORITY == 100

    def test_poc_request_priority_type(self):
        """POC_REQUEST_PRIORITY should be integer."""
        assert isinstance(POC_REQUEST_PRIORITY, int)


class TestV1IntegrationModuleStructure:
    """Tests for v1 module structure."""

    def test_v1_module_exists(self):
        """vllm.poc.v1 should be importable."""
        import vllm.poc.v1
        assert vllm.poc.v1 is not None

    def test_gpu_runner_module_exists(self):
        """vllm.poc.v1.gpu_runner should exist."""
        import vllm.poc.v1.gpu_runner
        assert vllm.poc.v1.gpu_runner is not None

    def test_async_engine_module_exists(self):
        """vllm.poc.v1.async_engine should exist."""
        import vllm.poc.v1.async_engine
        assert vllm.poc.v1.async_engine is not None

    def test_constants_module_exists(self):
        """vllm.poc.v1.constants should exist."""
        import vllm.poc.v1.constants
        assert vllm.poc.v1.constants is not None


class TestIntegrationImports:
    """Tests for imports used by integration."""

    def test_engine_core_request_importable(self):
        """EngineCoreRequest should be importable from vllm.v1.engine."""
        from vllm.v1.engine import EngineCoreRequest
        assert EngineCoreRequest is not None

    def test_engine_core_request_kind_importable(self):
        """EngineCoreRequestKind should be importable from vllm.v1.engine."""
        from vllm.v1.engine import EngineCoreRequestKind
        assert EngineCoreRequestKind is not None

    def test_sampling_params_importable(self):
        """SamplingParams should be importable."""
        from vllm.sampling_params import SamplingParams
        assert SamplingParams is not None

    def test_poc_params_importable(self):
        """PoCParams should be importable."""
        from vllm.poc.v1.params import PoCParams
        assert PoCParams is not None
