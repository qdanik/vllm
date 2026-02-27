"""vLLM Proof-of-Compute (PoC) Module."""

from vllm.poc.api.routes import router as poc_router

__all__ = ["poc_router"]
