import pytest
import torch

from vllm_latchmoe_cuda.core.waves import (
    ExactWavePlan,
    PairDescriptor,
    WaveDescriptor,
    plan_device_exact_waves,
    plan_exact_waves,
    plan_transfer_issue_order,
    plan_main_cache_waves,
    validate_pair_coverage,
)
from vllm_latchmoe_cuda.errors import PairIntegrityError
from vllm_latchmoe_cuda.offloader import WAVE_SLOTS_ENV, _wave_slot_count
from vllm_latchmoe_cuda.runner_adapter import _validate_piecewise_slot_capacity


def _routing_with_union(union: int, top_k: int = 8):
    experts = torch.arange(union, dtype=torch.int64)
    padding = (-union) % top_k
    if padding:
        repeats = (padding + union - 1) // union
        experts = torch.cat((experts, experts.repeat(repeats)[:padding]))
    ids = experts.view(-1, top_k)
    weights = torch.arange(1, ids.numel() + 1, dtype=torch.float32).view_as(ids)
    return ids, weights


@pytest.mark.parametrize("union", [1, 32, 33, 64, 127, 128])
def test_wave_plan_covers_every_pair_once(union: int):
    ids, weights = _routing_with_union(union)

    plan = plan_exact_waves(ids, weights, capacity=32)

    assert max(len(wave.experts) for wave in plan.waves) <= 32
    assert plan.all_pair_offsets() == tuple(range(ids.numel()))
    validate_pair_coverage(plan, expected_pairs=ids.numel())


def test_pair_descriptor_preserves_token_topk_and_weight():
    ids = torch.tensor([[5, 2], [9, 5]], dtype=torch.int64)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    plan = plan_exact_waves(ids, weights, capacity=2)
    pair = plan.pair(3)

    assert pair == PairDescriptor(
        pair_offset=3,
        token_index=1,
        topk_position=1,
        expert_id=5,
        weight=pytest.approx(0.4),
    )


def test_validation_rejects_duplicate_pair_assignment():
    pair = PairDescriptor(0, 0, 0, 1, 1.0)
    plan = ExactWavePlan(
        capacity=1,
        top_k=1,
        num_tokens=1,
        waves=(WaveDescriptor(0, (1,), (pair, pair)),),
        compute_order=(0,),
        issue_order=(0,),
    )

    with pytest.raises(PairIntegrityError, match="duplicate"):
        validate_pair_coverage(plan, expected_pairs=1)


def test_transfer_issue_order_does_not_change_compute_order():
    ids, weights = _routing_with_union(64)
    plan = plan_exact_waves(ids, weights, capacity=32)

    issue_order = plan_transfer_issue_order(
        plan.waves,
        ready_experts=frozenset(range(16)),
        h2d_bytes_by_wave={0: 400, 1: 1000},
    )

    assert plan.compute_order == (0, 1)
    assert issue_order == (1, 0)


def test_device_wave_plan_keeps_pair_descriptors_on_device():
    ids = torch.tensor([[5, 2], [9, 5]], dtype=torch.int64)
    weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    plan = plan_device_exact_waves(ids, weights, capacity=2, num_experts=16)

    assert plan.pair_count == ids.numel()
    assert tuple(wave.experts for wave in plan.waves) == ((2, 5), (9,))
    assert plan.waves[0].pair_offsets.tolist() == [0, 1, 3]
    assert plan.waves[0].token_indices.tolist() == [0, 0, 1]
    assert plan.waves[0].logical_ids.flatten().tolist() == [5, 2, 5]
    assert plan.waves[0].physical_ids.flatten().tolist() == [1, 0, 1]
    torch.testing.assert_close(
        plan.waves[0].pair_weights.flatten(), torch.tensor([0.1, 0.2, 0.4])
    )


def test_device_wave_plan_rejects_out_of_range_experts():
    ids = torch.tensor([[0, 16]], dtype=torch.int64)
    weights = torch.ones_like(ids, dtype=torch.float32)

    with pytest.raises(ValueError, match="invalid expert ids"):
        plan_device_exact_waves(ids, weights, capacity=2, num_experts=16)


def test_piecewise_finite_slots_cover_every_captured_routed_pair():
    assert (
        _validate_piecewise_slot_capacity(
            num_slots=64,
            num_experts=128,
            top_k=8,
            max_capture_size=8,
        )
        == 64
    )
    with pytest.raises(RuntimeError, match="required=64"):
        _validate_piecewise_slot_capacity(
            num_slots=32,
            num_experts=128,
            top_k=8,
            max_capture_size=8,
        )


def test_piecewise_full_capacity_does_not_require_capture_metadata():
    assert (
        _validate_piecewise_slot_capacity(
            num_slots=128,
            num_experts=128,
            top_k=0,
            max_capture_size=None,
        )
        == 128
    )


def test_wave_capacity_is_decoupled_from_main_graph_slots(monkeypatch):
    monkeypatch.delenv(WAVE_SLOTS_ENV, raising=False)
    assert _wave_slot_count(64) == 32
    assert _wave_slot_count(16) == 16

    monkeypatch.setenv(WAVE_SLOTS_ENV, "8")
    assert _wave_slot_count(64) == 8

    monkeypatch.setenv(WAVE_SLOTS_ENV, "65")
    with pytest.raises(ValueError, match=WAVE_SLOTS_ENV):
        _wave_slot_count(64)


def test_main_cache_waves_are_hit_first_and_capacity_bounded():
    specs = plan_main_cache_waves(
        active_experts=(0, 1, 2, 3, 4, 5, 6), capacity=4, hit_experts=(0, 1)
    )
    assert [(spec.wave_type, spec.experts) for spec in specs] == [
        ("hit", (0, 1)),
        ("miss", (2, 3)),
        ("miss", (4, 5, 6)),
    ]


def test_main_cache_wave_planner_does_not_duplicate_experts():
    specs = plan_main_cache_waves(
        active_experts=(0, 0, 1, 2, 3, 4), capacity=4, hit_experts=(0,)
    )
    assert tuple(expert for spec in specs for expert in spec.experts) == (
        0, 1, 2, 3, 4
    )


def test_full_hit_capacity_has_no_overlap_candidate():
    specs = plan_main_cache_waves(
        active_experts=(0, 1, 2, 3, 4), capacity=4, hit_experts=(0, 1, 2, 3)
    )
    assert specs[0].overlap_candidate is False
    assert specs[1].overlap_candidate is False


def test_partial_hit_marks_only_capacity_safe_next_wave_as_overlap_candidate():
    specs = plan_main_cache_waves(
        active_experts=(0, 1, 2, 3),
        capacity=4,
        hit_experts=(0, 1),
    )

    assert [(spec.wave_type, spec.experts) for spec in specs] == [
        ("hit", (0, 1)),
        ("miss", (2, 3)),
    ]
    assert specs[0].overlap_candidate is True
    assert specs[1].overlap_candidate is False


def test_partial_hit_splits_first_miss_wave_to_fit_idle_slots():
    specs = plan_main_cache_waves(
        active_experts=(0, 1, 2, 3, 4, 5),
        capacity=4,
        hit_experts=(0, 1),
    )

    assert [(spec.wave_type, spec.experts) for spec in specs] == [
        ("hit", (0, 1)),
        ("miss", (2, 3)),
        ("miss", (4, 5)),
    ]
    assert specs[0].overlap_candidate is True
