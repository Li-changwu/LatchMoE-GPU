import json

import pytest

from vllm_latchmoe_cuda.errors import PlanValidationError, ResidencyBudgetError
from vllm_latchmoe_cuda.residency_plan import (
    build_residency_plan,
    deserialize_residency_plan,
    serialize_residency_plan,
)


QWEN3_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}
GLM_CONFIG = {
    "model_type": "glm4_moe_lite",
    "hidden_size": 2048,
    "moe_intermediate_size": 1536,
    "n_routed_experts": 64,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 47,
    "dense_layer_ids": [0],
    "torch_dtype": "bfloat16",
}


def _qwen_plan(gib=13.5):
    return build_residency_plan(
        gib,
        QWEN3_CONFIG,
        max_capture_size=1,
        top_k=8,
        device_total_bytes=2 << 40,
        kv_reserve_bytes=512 << 20,
    )


def test_qwen_default_is_complete_layer_midpoint_partition():
    plan = _qwen_plan()

    assert plan.selection_strategy == "midpoint_stratified_v1"
    assert plan.offloaded_layer_ids == tuple(range(2, 48, 4))
    assert set(plan.offloaded_layer_ids).isdisjoint(plan.resident_layer_ids)
    assert tuple(sorted(plan.offloaded_layer_ids + plan.resident_layer_ids)) == (
        plan.eligible_layer_ids
    )
    assert plan.effective_offloaded_bytes >= plan.requested_offload_bytes
    assert plan.effective_num_slots == 8


def test_glm_dense_layer_is_filtered_before_selection():
    plan = build_residency_plan(
        13.5,
        GLM_CONFIG,
        max_capture_size=1,
        top_k=4,
        device_total_bytes=2 << 40,
        kv_reserve_bytes=0,
    )

    assert plan.eligible_layer_ids == tuple(range(1, 47))
    assert plan.offloaded_layer_ids == (
        2, 6, 10, 14, 18, 22, 25, 29, 33, 37, 41, 45,
    )
    assert 0 not in plan.resident_layer_ids


def test_heterogeneous_bytes_use_shortest_complete_prefix():
    config = {
        "model_type": "synthetic_moe",
        "num_hidden_layers": 3,
        "moe_layer_ids": [0, 1, 2],
        "num_experts": 2,
    }
    metadata = (
        {"layer_id": 0, "routed_expert_bytes": 4},
        {"layer_id": 1, "routed_expert_bytes": 8},
        {"layer_id": 2, "routed_expert_bytes": 16},
    )

    plan = build_residency_plan(
        10 / (1 << 30),
        config,
        max_capture_size=1,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=metadata,
    )

    assert plan.selection_strategy == "ordered_prefix_bytes_v1"
    assert plan.offloaded_layer_ids == (0, 1)
    assert plan.effective_offloaded_bytes == 12


def test_zero_and_capped_budgets_are_explicit():
    assert _qwen_plan(0).offloaded_layer_ids == ()
    capped = _qwen_plan(1000)
    assert capped.offloaded_layer_ids == tuple(range(48))
    assert capped.resident_layer_ids == ()
    assert capped.ledger["capped_to_eligible_layers"] is True


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("selection_strategy", "legacy"),
        ("eligible_layer_ids", list(range(47))),
        ("offloaded_layer_ids", list(range(12))),
        ("plan_id", "0" * 64),
    ],
)
def test_round_trip_preserves_digest_and_rejects_tampering(field, replacement):
    plan = _qwen_plan()
    restored = deserialize_residency_plan(serialize_residency_plan(plan))
    assert restored.plan_id == plan.plan_id
    assert restored.eligible_layer_ids == plan.eligible_layer_ids
    document = restored.to_jsonable()
    document[field] = replacement

    with pytest.raises(PlanValidationError):
        deserialize_residency_plan(json.dumps(document))


def test_shared_bytes_do_not_enter_dynamic_slot_budget():
    config = {"model_type": "synthetic_moe", "moe_layer_ids": [0], "num_experts": 4}
    plan = build_residency_plan(
        96 / (1 << 30),
        config,
        max_capture_size=1,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=(
            {"layer_id": 0, "routed_expert_bytes": 96, "shared_expert_bytes": 1000},
        ),
    )
    assert plan.main_slot_cache_bytes == 24


def test_unknown_architecture_without_measured_metadata_fails_closed():
    with pytest.raises(PlanValidationError, match="metadata"):
        build_residency_plan(
            1,
            {"model_type": "unknown", "num_hidden_layers": 2, "num_experts": 4},
            max_capture_size=1,
            top_k=1,
            device_total_bytes=1 << 40,
            kv_reserve_bytes=0,
        )


def test_slot_floor_must_produce_positive_hbm_savings():
    with pytest.raises(ResidencyBudgetError, match="does not save HBM"):
        build_residency_plan(
            1,
            QWEN3_CONFIG,
            max_capture_size=16,
            top_k=8,
            device_total_bytes=2 << 40,
            kv_reserve_bytes=0,
        )

