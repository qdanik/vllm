"""PoC forward runtime orchestration for the v1 GPU model runner."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class _PoCRuntimeState:
    token_mask: Any
    householder_vectors: torch.Tensor | None = None
    householder_block_hash: str | None = None
    layer_hooks: Any = None


class PoCForwardRuntime:
    """Encapsulates PoC-specific forward-path setup and teardown."""

    def __init__(self, runner: Any) -> None:
        self.runner = runner
        self._state: _PoCRuntimeState | None = None

    def initialize(self) -> None:
        self._state = _PoCRuntimeState(
            token_mask=self.runner._make_buffer(
                self.runner.max_num_tokens,
                dtype=torch.bool,
            )
        )

    def _require_state(self) -> _PoCRuntimeState:
        if self._state is None:
            raise RuntimeError("PoCForwardRuntime is not initialized")
        return self._state

    def update_token_mask(
        self,
        *,
        scheduler_output: Any,
        total_num_scheduled_tokens: int,
        req_indices: Any,
        num_reqs: int,
    ) -> None:
        state = self._require_state()
        req_ids = self.runner.input_batch.req_ids
        is_poc_req = np.fromiter(
            (
                (
                    (self.runner.requests.get(rid) is not None)
                    and (self.runner.requests[rid].poc_params is not None)
                )
                for rid in req_ids
            ),
            dtype=np.bool_,
            count=num_reqs,
        )
        state.token_mask.np[:total_num_scheduled_tokens] = is_poc_req[req_indices]
        state.token_mask.np[total_num_scheduled_tokens:].fill(False)
        state.token_mask.copy_to_gpu(total_num_scheduled_tokens)

    @property
    def token_mask_gpu(self) -> torch.Tensor:
        state = self._require_state()
        return state.token_mask.gpu

    def batch_has_poc(self, scheduler_output: Any) -> bool:
        if getattr(scheduler_output, "poc_req_ids", None):
            return True
        from vllm.poc.v1.gpu_model_runner_integration import batch_has_poc

        return batch_has_poc(scheduler_output, self.runner.requests)

    def fill_inputs_embeds(self, scheduler_output: Any) -> None:
        from vllm.poc.v1.gpu_model_runner_integration import fill_poc_inputs_embeds

        return fill_poc_inputs_embeds(
            scheduler_output,
            self.runner.requests,
            self.runner.input_batch,
            self.runner.inputs_embeds.gpu,
            self.runner.is_token_ids.gpu,
            self.runner.device,
            self.runner.dtype,
            self.runner.pp_group.is_first_rank,
            self.runner.model_config,
        )

    def extract_results(
        self,
        scheduler_output: Any,
        sample_hidden_states: torch.Tensor,
    ) -> dict[str, dict] | None:
        from vllm.poc.v1.gpu_model_runner_integration import extract_poc_results

        return extract_poc_results(
            scheduler_output,
            self.runner.requests,
            self.runner.input_batch,
            sample_hidden_states,
            self.runner.device,
        )

    def get_poc_block_hash(self, scheduler_output: Any) -> str | None:
        block_hash: str | None = None
        saw_multiple = False
        for req_id in scheduler_output.num_scheduled_tokens:
            req_state = self.runner.requests.get(req_id)
            if req_state is None or req_state.poc_params is None:
                continue
            bh = req_state.poc_params.block_hash
            if block_hash is None:
                block_hash = bh
            elif bh != block_hash:
                saw_multiple = True

        if saw_multiple and block_hash is not None:
            logger.warning(
                "PoC batch contains multiple block_hash values; "
                "in-graph transforms will use the first one (%s)",
                block_hash,
            )

        return block_hash

    def _has_poc_householder_context(self, model: Any) -> bool:
        return hasattr(model, "set_poc_householder_context")

    def get_poc_in_graph_target_model(self) -> torch.nn.Module | None:
        raw_model = self.runner.get_model()
        candidate = getattr(raw_model, "model", None)
        if candidate is not None and self._has_poc_householder_context(candidate):
            return candidate
        if self._has_poc_householder_context(raw_model):
            return raw_model

        candidate = getattr(self.runner.model, "model", None)
        if candidate is not None and hasattr(candidate, "set_poc_householder_context"):
            return candidate
        if hasattr(self.runner.model, "set_poc_householder_context"):
            return self.runner.model
        return None

    def ensure_poc_householder_vectors(
        self,
        *,
        block_hash: str,
        num_layers: int,
    ) -> torch.Tensor:
        from vllm.poc.core.transforms import generate_householder_vector

        state = self._require_state()
        hidden_size = self.runner.model_config.get_hidden_size()
        dtype = self.runner.dtype

        if (
            state.householder_vectors is None
            or state.householder_vectors.device != self.runner.device
            or state.householder_vectors.dtype != dtype
            or state.householder_vectors.shape != (num_layers, hidden_size)
        ):
            state.householder_vectors = torch.empty(
                (num_layers, hidden_size), device=self.runner.device, dtype=dtype
            )
            state.householder_block_hash = None

        if state.householder_block_hash != block_hash:
            assert state.householder_vectors is not None
            for layer_idx in range(num_layers):
                seed_str = f"{block_hash}_layer_{layer_idx}_householder"
                v = generate_householder_vector(
                    seed_str, hidden_size, self.runner.device
                )
                state.householder_vectors[layer_idx].copy_(v.to(dtype))
            state.householder_block_hash = block_hash

        return state.householder_vectors

    def ensure_poc_layer_hooks(self, *, block_hash: str) -> bool:
        from vllm.poc.core.layer_hooks import LayerHouseholderHook

        state = self._require_state()
        hook = state.layer_hooks
        if (
            hook is not None
            and getattr(hook, "block_hash", None) == block_hash
            and hook.num_layers > 0
        ):
            return True

        if hook is not None:
            hook.detach()

        raw_model = self.runner.get_model()
        hidden_size = self.runner.model_config.get_hidden_size()
        new_hook = LayerHouseholderHook(
            raw_model, block_hash, self.runner.device, hidden_size
        )
        new_hook.attach()
        state.layer_hooks = new_hook
        return new_hook.num_layers > 0

    def prepare_forward(
        self,
        *,
        has_poc: bool,
        scheduler_output: Any,
        num_tokens_padded: int,
    ) -> tuple[torch.nn.Module | None, bool]:
        poc_in_graph_target: torch.nn.Module | None = None
        use_poc_layer_hooks = False
        if not has_poc:
            return poc_in_graph_target, use_poc_layer_hooks

        block_hash = self.get_poc_block_hash(scheduler_output)
        if block_hash is None:
            return poc_in_graph_target, use_poc_layer_hooks

        req_ids = self.runner.input_batch.req_ids
        apply_all = True
        for rid in req_ids:
            rs = self.runner.requests.get(rid)
            if rs is None or rs.poc_params is None:
                apply_all = False
                break

        if apply_all:
            try:
                use_poc_layer_hooks = self.ensure_poc_layer_hooks(block_hash=block_hash)
            except Exception:
                logger.exception("Failed to set PoC layer hooks")
                use_poc_layer_hooks = False

        if use_poc_layer_hooks:
            return None, True

        poc_in_graph_target = self.get_poc_in_graph_target_model()
        if poc_in_graph_target is None or not hasattr(
            poc_in_graph_target, "clear_poc_householder_context"
        ):
            return None, False

        num_layers = getattr(
            getattr(poc_in_graph_target, "config", None),
            "num_hidden_layers",
            len(getattr(poc_in_graph_target, "layers", ())),
        )
        if num_layers <= 0:
            return None, False

        householder_vectors = self.ensure_poc_householder_vectors(
            block_hash=block_hash,
            num_layers=num_layers,
        )

        valid_tokens = scheduler_output.total_num_scheduled_tokens
        if (not apply_all) and valid_tokens < num_tokens_padded:
            self.token_mask_gpu[valid_tokens:num_tokens_padded].fill_(False)

        try:
            poc_in_graph_target.set_poc_householder_context(
                householder_vectors=householder_vectors,
                token_mask=None if apply_all else self.token_mask_gpu,
                apply_all=apply_all,
            )
        except Exception:
            logger.exception("Failed to set in-graph PoC Householder context")
            return None, False

        return poc_in_graph_target, False

    def forward_hooks_context(self, *, has_poc: bool, use_poc_layer_hooks: bool):
        if not (has_poc and use_poc_layer_hooks):
            return nullcontext()

        from vllm.poc.core.layer_hooks import poc_forward_context

        return poc_forward_context()

    def cleanup_after_forward(self, poc_in_graph_target: torch.nn.Module | None) -> None:
        if poc_in_graph_target is None:
            return
        try:
            poc_in_graph_target.clear_poc_householder_context()
        except Exception:
            logger.exception("Failed to clear in-graph PoC context")