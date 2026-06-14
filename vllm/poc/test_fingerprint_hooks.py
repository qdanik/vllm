# SPDX-License-Identifier: Apache-2.0
"""Integration smoke test for the routing fingerprint hook.

Builds a tiny fake module tree with FusedMoE-like leaves and asserts the
forward_pre_hook fires (only while a PoC forward is active) and produces routing
decisions with the expected site mapping. No GPU and no real model needed.
"""
import torch

from vllm.poc.fingerprint.logit_capture import capture_seeded_logits
from vllm.poc.fingerprint.schema import LOGIT_KIND, ROUTING_KIND
from vllm.poc.fingerprint_hooks import RoutingFingerprintHook
from vllm.poc.layer_hooks import poc_forward_context


class FakeFusedMoE(torch.nn.Module):
    """Minimal stand-in for FusedMoE: forward(hidden_states, router_logits)."""

    def __init__(self, top_k: int, is_internal_router: bool = False):
        super().__init__()
        self.top_k = top_k
        self.is_internal_router = is_internal_router

    def forward(self, hidden_states, router_logits, input_ids=None):
        # The real FusedMoE returns mixed hidden states; the test only cares
        # that the pre-hook saw router_logits, so return hidden_states as-is.
        return hidden_states


class FakeModelWithMoE(torch.nn.Module):
    def __init__(self, num_moe: int, top_k: int):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [FakeFusedMoE(top_k=top_k) for _ in range(num_moe)]
        )

    def forward(self, hidden_states, router_logits):
        for block in self.blocks:
            hidden_states = block(hidden_states, router_logits)
        return hidden_states


def _is_fake_moe(module):
    return isinstance(module, FakeFusedMoE)


def test_hook_finds_and_installs_on_all_fake_moe():
    model = FakeModelWithMoE(num_moe=3, top_k=2)
    hook = RoutingFingerprintHook(block_hash="bh")
    hook._setup(model, is_moe=_is_fake_moe)
    assert hook.num_layers == 3
    hook.detach()
    assert hook.num_layers == 0


def test_hook_no_capture_when_poc_inactive():
    model = FakeModelWithMoE(num_moe=2, top_k=2)
    hook = RoutingFingerprintHook(block_hash="bh")
    hook._setup(model, is_moe=_is_fake_moe)
    hook.set_seq_len(2)

    # batch_size=1, seq_len=2 -> 2 token rows; 4 experts.
    hidden = torch.zeros(2, 8)
    router_logits = torch.randn(2, 4)
    # No poc_forward_context -> hook must not capture.
    model(hidden, router_logits)
    assert hook.drain() == []
    hook.detach()


def test_hook_captures_routing_decisions_under_poc_context():
    num_moe = 3
    top_k = 2
    seq_len = 2
    batch_size = 1
    model = FakeModelWithMoE(num_moe=num_moe, top_k=top_k)
    hook = RoutingFingerprintHook(block_hash="bh")
    hook._setup(model, is_moe=_is_fake_moe)
    hook.set_seq_len(seq_len)

    num_tokens = batch_size * seq_len
    num_experts = 4
    # Graded router logits so top-2 (and the margin gap top-2 minus top-3) is
    # well defined: row r ranks expert (r % num_experts) highest, descending.
    router_logits = torch.zeros(num_tokens, num_experts)
    for row in range(num_tokens):
        for expert in range(num_experts):
            rank = (expert - row) % num_experts
            router_logits[row, expert] = float(num_experts - rank)
    hidden = torch.zeros(num_tokens, 8)

    with poc_forward_context():
        model(hidden, router_logits)

    decisions = hook.drain()
    # One decision per (moe_layer, token_row).
    assert len(decisions) == num_moe * num_tokens
    # Drain clears the buffer.
    assert hook.drain() == []

    # Check the site mapping and ids for one entry.
    nonce_idx, position_idx, decision = decisions[0]
    assert decision.kind == ROUTING_KIND
    assert nonce_idx == 0
    assert position_idx == 0
    assert decision.site_id[1] == 0  # position component
    assert decision.topk_ids[0] == 0  # row 0 -> expert 0
    assert decision.margin > 0.0
    hook.detach()


def test_hook_warns_on_internal_router():
    # vLLM's logger does not propagate to pytest's caplog, so assert on the
    # hook's own warned-once flag instead of captured log records.
    model = torch.nn.Module()
    model.moe = FakeFusedMoE(top_k=2, is_internal_router=True)
    hook = RoutingFingerprintHook(block_hash="bh")
    assert hook._warned_internal_router is False
    hook._setup(model, is_moe=_is_fake_moe)
    assert hook._warned_internal_router is True
    hook.detach()


class FakeModelWithLMHead:
    """Exposes compute_logits like a vLLM model for logit-capture testing."""

    def __init__(self, vocab: int):
        # A fixed projection so logits are deterministic per hidden row.
        self.vocab = vocab

    def compute_logits(self, hidden_states):
        # hidden_states: [rows, hidden]; produce [rows, vocab] where the argmax
        # token equals the integer in hidden_states[:, 0].
        rows = hidden_states.shape[0]
        logits = torch.zeros(rows, self.vocab)
        for row in range(rows):
            target_token = int(hidden_states[row, 0].item()) % self.vocab
            logits[row, target_token] = 100.0
            logits[row, (target_token + 1) % self.vocab] = 10.0
        return logits


def test_capture_seeded_logits():
    batch_size, seq_len, hidden = 2, 4, 3
    hidden_states_3d = torch.zeros(batch_size, seq_len, hidden)
    # Encode the desired argmax token id into channel 0 at each position.
    hidden_states_3d[0, 1, 0] = 5.0
    hidden_states_3d[0, 3, 0] = 7.0
    hidden_states_3d[1, 2, 0] = 9.0

    model = FakeModelWithLMHead(vocab=20)
    seeded = {0: [1, 3], 1: [2]}
    result = capture_seeded_logits(model, hidden_states_3d, seeded, top_k=1)

    assert set(result.keys()) == {0, 1}
    assert len(result[0]) == 2
    assert len(result[1]) == 1
    nonce0_first = result[0][0]
    assert nonce0_first.kind == LOGIT_KIND
    assert nonce0_first.site_id == [1]
    assert nonce0_first.topk_ids == [5]  # token 5 from hidden value 5.0
    assert result[1][0].topk_ids == [9]
    assert result[1][0].site_id == [2]


def test_capture_seeded_logits_empty_positions():
    model = FakeModelWithLMHead(vocab=10)
    hidden_states_3d = torch.zeros(1, 2, 3)
    result = capture_seeded_logits(model, hidden_states_3d, {0: []}, top_k=1)
    assert result == {0: []}
