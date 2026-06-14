# SPDX-License-Identifier: Apache-2.0
"""Seeded LM-head logit capture for the PoC discrete fingerprint.

Gathers hidden states at the seeded ``(nonce, position)`` rows, runs them
through ``model.compute_logits`` once, and turns each resulting logit row into a
logit :class:`~vllm.poc.fingerprint.schema.CapturedDecision`. Only a few seeded
positions are captured per nonce, so the extra LM-head matmul is cheap relative
to the forward (architecture.md Section 5.4).

The pure numeric core (top-k + margin) lives in
:mod:`vllm.poc.fingerprint.decisions`; this module is the thin model-coupled
wrapper.
"""
import torch

from vllm.poc.fingerprint.decisions import logits_row_to_decision
from vllm.poc.fingerprint.schema import CapturedDecision


def capture_seeded_logits(
    model,
    hidden_states_3d: torch.Tensor,
    seeded_positions_per_nonce: dict[int, list[int]],
    top_k: int,
) -> dict[int, list[CapturedDecision]]:
    """Capture top-``k`` logit decisions at seeded positions for each nonce.

    Args:
        model: The model exposing ``compute_logits(hidden_states) -> logits``
            (a ``[rows, vocab]`` tensor).
        hidden_states_3d: ``[batch_size, seq_len, hidden]`` final hidden states,
            indexed by ``(nonce_idx, position_idx)``.
        seeded_positions_per_nonce: Map ``nonce_idx -> [position_idx, ...]`` of
            the seeded logit positions for that nonce (from
            :func:`vllm.poc.fingerprint.site_selection.pick_logit_positions`).
        top_k: Number of token ids to keep per position.

    Returns:
        Map ``nonce_idx -> [CapturedDecision, ...]`` with one logit decision per
        seeded position (``site_id == [position_idx]``). Empty when no positions
        are seeded.
    """
    # Flatten the requested (nonce, position) rows into one batch so the LM head
    # runs a single matmul over only the seeded rows (cheap).
    gather_nonce_idx: list[int] = []
    gather_position_idx: list[int] = []
    for nonce_idx, positions in seeded_positions_per_nonce.items():
        for position_idx in positions:
            gather_nonce_idx.append(nonce_idx)
            gather_position_idx.append(position_idx)

    decisions_per_nonce: dict[int, list[CapturedDecision]] = {
        nonce_idx: [] for nonce_idx in seeded_positions_per_nonce
    }
    if not gather_nonce_idx:
        return decisions_per_nonce

    device = hidden_states_3d.device
    nonce_index_tensor = torch.tensor(
        gather_nonce_idx, dtype=torch.long, device=device
    )
    position_index_tensor = torch.tensor(
        gather_position_idx, dtype=torch.long, device=device
    )
    selected_rows = hidden_states_3d[nonce_index_tensor, position_index_tensor]

    logits = model.compute_logits(selected_rows)
    if logits is None:
        return decisions_per_nonce

    for row, (nonce_idx, position_idx) in enumerate(
        zip(gather_nonce_idx, gather_position_idx)
    ):
        decision = logits_row_to_decision(logits[row], top_k, position_idx)
        decisions_per_nonce[nonce_idx].append(decision)

    return decisions_per_nonce
