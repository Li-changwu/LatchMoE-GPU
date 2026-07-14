from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from ..errors import IllegalSlotTransitionError, StaleMappingError
from .expert_key import ExpertKey


class SlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    COMPUTING = "computing"


@dataclass
class ExpertSlot:
    slot_id: int
    key: ExpertKey | None = None
    state: SlotState = SlotState.EMPTY
    generation: int = 0
    last_used: int = 0


@dataclass(frozen=True)
class SlotLease:
    slot_id: int
    key: ExpertKey
    generation: int


@dataclass(frozen=True)
class ComputeHandle:
    leases: tuple[SlotLease, ...]


class ExpertSlotBank:
    def __init__(self, num_slots: int):
        if num_slots <= 0:
            raise ValueError("num_slots must be positive")
        self.slots = [ExpertSlot(slot_id=value) for value in range(num_slots)]
        self._clock = 0

    def _next_clock(self) -> int:
        self._clock += 1
        return self._clock

    def _slot(self, slot_id: int) -> ExpertSlot:
        try:
            return self.slots[slot_id]
        except IndexError as exc:
            raise IndexError(f"invalid slot_id={slot_id}") from exc

    def begin_load(self, key: ExpertKey, *, slot_id: int) -> SlotLease:
        slot = self._slot(slot_id)
        if slot.state not in (SlotState.EMPTY, SlotState.READY):
            raise IllegalSlotTransitionError(
                f"slot {slot_id} cannot begin loading while {slot.state.value}"
            )
        slot.generation += 1
        slot.key = key
        slot.state = SlotState.LOADING
        slot.last_used = self._next_clock()
        return SlotLease(slot_id=slot_id, key=key, generation=slot.generation)

    def mark_ready(self, slot_id: int, generation: int) -> SlotLease:
        slot = self._slot(slot_id)
        self.validate_generation(slot_id, generation)
        if slot.state is not SlotState.LOADING or slot.key is None:
            raise IllegalSlotTransitionError(
                f"slot {slot_id} cannot become ready from {slot.state.value}"
            )
        slot.state = SlotState.READY
        slot.last_used = self._next_clock()
        return SlotLease(slot_id, slot.key, slot.generation)

    def begin_compute(self, slot_ids: Iterable[int]) -> ComputeHandle:
        unique_ids = tuple(dict.fromkeys(int(value) for value in slot_ids))
        leases: list[SlotLease] = []
        for slot_id in unique_ids:
            slot = self._slot(slot_id)
            if slot.state is not SlotState.READY or slot.key is None:
                raise IllegalSlotTransitionError(
                    f"slot {slot_id} cannot compute while {slot.state.value}"
                )
            leases.append(SlotLease(slot_id, slot.key, slot.generation))
        for lease in leases:
            slot = self._slot(lease.slot_id)
            slot.state = SlotState.COMPUTING
            slot.last_used = self._next_clock()
        return ComputeHandle(tuple(leases))

    def end_compute(self, handle: ComputeHandle) -> None:
        for lease in handle.leases:
            slot = self._slot(lease.slot_id)
            self.validate_generation(lease.slot_id, lease.generation)
            if slot.state is not SlotState.COMPUTING:
                raise IllegalSlotTransitionError(
                    f"slot {lease.slot_id} cannot end compute from {slot.state.value}"
                )
        for lease in handle.leases:
            slot = self._slot(lease.slot_id)
            slot.state = SlotState.READY
            slot.last_used = self._next_clock()

    def evict(self, slot_id: int) -> None:
        slot = self._slot(slot_id)
        if slot.state is not SlotState.READY:
            raise IllegalSlotTransitionError(
                f"slot {slot_id} cannot be evicted while {slot.state.value}"
            )
        slot.key = None
        slot.state = SlotState.EMPTY
        slot.last_used = self._next_clock()

    def touch(self, slot_id: int) -> None:
        slot = self._slot(slot_id)
        if slot.state not in (SlotState.READY, SlotState.COMPUTING):
            raise IllegalSlotTransitionError(
                f"slot {slot_id} cannot be touched while {slot.state.value}"
            )
        slot.last_used = self._next_clock()

    def validate_generation(self, slot_id: int, expected: int) -> None:
        actual = self._slot(slot_id).generation
        if actual != expected:
            raise StaleMappingError(
                f"slot generation mismatch: slot={slot_id}, "
                f"expected={expected}, actual={actual}"
            )

    def ready_mapping(self) -> dict[ExpertKey, int]:
        return {
            slot.key: slot.slot_id
            for slot in self.slots
            if slot.key is not None
            and slot.state in (SlotState.READY, SlotState.COMPUTING)
        }

    def find(self, key: ExpertKey) -> ExpertSlot | None:
        return next(
            (
                slot
                for slot in self.slots
                if slot.key == key
                and slot.state in (SlotState.READY, SlotState.COMPUTING)
            ),
            None,
        )
