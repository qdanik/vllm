# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from dataclasses import dataclass

from vllm.v1.engine import PoCParams
from vllm.v1.request import RequestStatus


@dataclass(eq=False)
class PoCRequest:
    """Internal PoC request representation for the v1 scheduler.

    Unlike the standard Request:
    - has no tokens/embeds
    - must not allocate/use KV cache
    - executes as a single-step forward
    """

    request_id: str
    client_index: int
    arrival_time: float
    priority: int
    poc_params: PoCParams

    status: RequestStatus = RequestStatus.WAITING

    def __lt__(self, other: "PoCRequest") -> bool:
        if self.priority != other.priority:
            return self.priority < other.priority
        if self.arrival_time != other.arrival_time:
            return self.arrival_time < other.arrival_time
        if self.request_id != other.request_id:
            return self.request_id < other.request_id
        return id(self) < id(other)

    def is_finished(self) -> bool:
        return RequestStatus.is_finished(self.status)
