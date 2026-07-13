from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if self.expert_id < 0:
            raise ValueError("expert_id must be non-negative")

    def to_jsonable(self) -> dict[str, int]:
        return {"layer_id": self.layer_id, "expert_id": self.expert_id}

    @classmethod
    def from_jsonable(cls, value: dict[str, Any]) -> ExpertKey:
        return cls(layer_id=int(value["layer_id"]), expert_id=int(value["expert_id"]))

