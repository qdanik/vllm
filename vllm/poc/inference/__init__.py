"""PoC inference integration with vLLM model runner and layer hooks."""

from vllm.poc.inference.layer_hooks import (
    LayerHouseholderHook,
    is_poc_forward_active,
    poc_forward_context,
)
from vllm.poc.inference.model_runner import execute_poc_forward

__all__ = [
    "execute_poc_forward",
    "LayerHouseholderHook",
    "poc_forward_context",
    "is_poc_forward_active",
]
