from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from vllm.poc.constants import (
    DEFAULT_DIST_THRESHOLD,
    DEFAULT_FRAUD_THRESHOLD,
    DEFAULT_K_DIM,
    DEFAULT_P_MISMATCH,
)
from vllm.poc.protocol.schemas import ArtifactSchema


class PoCParamsModel(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str
    seq_len: int
    k_dim: int = DEFAULT_K_DIM


class PoCInitGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    group_id: int = 0
    n_groups: int = 1
    params: PoCParamsModel
    url: str | None = None


@dataclass
class NonceIterator:
    """Iterator for nonces with multi-node and multi-group support."""

    node_id: int
    n_nodes: int
    group_id: int
    n_groups: int
    _current_x: int = 0

    def __iter__(self):
        return self

    def __next__(self) -> int:
        offset = self.node_id + self.group_id * self.n_nodes
        step = self.n_groups * self.n_nodes
        value = offset + self._current_x * step
        self._current_x += 1
        return value

    def take(self, n: int) -> list[int]:
        return [next(self) for _ in range(n)]


class ValidationModel(BaseModel):
    artifacts: list[ArtifactSchema]


class StatTestModel(BaseModel):
    dist_threshold: float = DEFAULT_DIST_THRESHOLD
    p_mismatch: float = DEFAULT_P_MISMATCH
    fraud_threshold: float = DEFAULT_FRAUD_THRESHOLD


class PoCGenerateRequest(BaseModel):
    block_hash: str
    block_height: int
    public_key: str
    node_id: int
    node_count: int
    nonces: list[int]
    params: PoCParamsModel
    wait: bool = False
    url: str | None = None
    validation: ValidationModel | None = None
    stat_test: StatTestModel | None = None
