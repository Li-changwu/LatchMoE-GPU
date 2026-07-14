from __future__ import annotations

from dataclasses import dataclass

from ..errors import NoEvictableSlotError
from .slots import ExpertSlotBank, SlotState


@dataclass(frozen=True)
class LruPolicy:
    def choose(self, bank: ExpertSlotBank, *, excluded: set[int] | None = None) -> int:
        excluded = excluded or set()
        empty = [
            slot
            for slot in bank.slots
            if slot.slot_id not in excluded and slot.state is SlotState.EMPTY
        ]
        if empty:
            return min(empty, key=lambda slot: slot.slot_id).slot_id
        ready = [
            slot
            for slot in bank.slots
            if slot.slot_id not in excluded and slot.state is SlotState.READY
        ]
        if ready:
            return min(ready, key=lambda slot: (slot.last_used, slot.slot_id)).slot_id
        states = ",".join(slot.state.value for slot in bank.slots)
        raise NoEvictableSlotError(
            f"no EMPTY or READY slot is evictable: states=[{states}]"
        )
