"""PoC generation configuration and state management."""

from dataclasses import dataclass
from enum import Enum

from vllm.poc.constants import DEFAULT_K_DIM


class PoCState(Enum):
    """State of PoC generation process."""

    IDLE = "IDLE"
    GENERATING = "GENERATING"
    STOPPED = "STOPPED"


@dataclass
class PoCConfig:
    """Configuration for a PoC generation round."""

    block_hash: str
    block_height: int
    public_key: str
    node_id: int = 0
    node_count: int = 1
    seq_len: int = 256
    k_dim: int = DEFAULT_K_DIM
    callback_url: str | None = None
    group_id: int = 0
    n_groups: int = 1
