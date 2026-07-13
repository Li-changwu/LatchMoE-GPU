import pytest

from vllm_latchmoe_cuda.core.expert_key import ExpertKey
from vllm_latchmoe_cuda.core.slots import ExpertSlotBank, SlotState
from vllm_latchmoe_cuda.errors import (
    IllegalSlotTransitionError,
    StaleMappingError,
)


def _ready_slot(bank: ExpertSlotBank, slot_id: int, key: ExpertKey):
    lease = bank.begin_load(key, slot_id=slot_id)
    bank.mark_ready(slot_id, lease.generation)
    return lease


def test_slot_follows_load_compute_lifecycle():
    bank = ExpertSlotBank(2)
    lease = _ready_slot(bank, 0, ExpertKey(3, 7))

    compute = bank.begin_compute((0,))
    assert bank.slots[0].state is SlotState.COMPUTING
    bank.end_compute(compute)

    assert bank.slots[0].state is SlotState.READY
    assert bank.ready_mapping() == {ExpertKey(3, 7): 0}
    bank.validate_generation(0, lease.generation)


def test_loading_slot_is_not_visible_in_ready_mapping():
    bank = ExpertSlotBank(1)
    bank.begin_load(ExpertKey(3, 1), slot_id=0)

    assert bank.ready_mapping() == {}


def test_computing_slot_cannot_begin_load():
    bank = ExpertSlotBank(1)
    _ready_slot(bank, 0, ExpertKey(3, 1))
    bank.begin_compute((0,))

    with pytest.raises(IllegalSlotTransitionError, match="computing"):
        bank.begin_load(ExpertKey(3, 2), slot_id=0)


def test_generation_changes_on_reuse_and_rejects_stale_snapshot():
    bank = ExpertSlotBank(1)
    first = _ready_slot(bank, 0, ExpertKey(3, 1))
    bank.evict(0)
    second = _ready_slot(bank, 0, ExpertKey(3, 2))

    assert second.generation > first.generation
    with pytest.raises(StaleMappingError, match="generation"):
        bank.validate_generation(0, first.generation)

