import pytest

from vllm_latchmoe_cuda.core.expert_key import ExpertKey
from vllm_latchmoe_cuda.core.policy import LruPolicy
from vllm_latchmoe_cuda.core.slots import ExpertSlotBank
from vllm_latchmoe_cuda.errors import NoEvictableSlotError


def _ready(bank: ExpertSlotBank, slot_id: int, expert_id: int):
    lease = bank.begin_load(ExpertKey(3, expert_id), slot_id=slot_id)
    bank.mark_ready(slot_id, lease.generation)


def test_lru_prefers_empty_then_oldest_ready_slot():
    bank = ExpertSlotBank(2)
    policy = LruPolicy()

    assert policy.choose(bank) == 0
    _ready(bank, 0, 1)
    assert policy.choose(bank) == 1
    _ready(bank, 1, 2)
    bank.touch(0)
    assert policy.choose(bank) == 1


def test_lru_never_evicts_loading_or_computing():
    bank = ExpertSlotBank(2)
    _ready(bank, 0, 1)
    bank.begin_compute((0,))
    bank.begin_load(ExpertKey(3, 2), slot_id=1)

    with pytest.raises(NoEvictableSlotError, match="no EMPTY or READY"):
        LruPolicy().choose(bank)


def test_lru_honors_explicit_exclusions():
    bank = ExpertSlotBank(1)
    _ready(bank, 0, 1)

    with pytest.raises(NoEvictableSlotError):
        LruPolicy().choose(bank, excluded={0})
