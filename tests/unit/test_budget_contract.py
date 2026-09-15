import pytest

from benchmark.scripts.verify_budget_contract import (
    BudgetContractError,
    validate_profile_events,
    validate_runtime_ledger,
)
from vllm_latchmoe_cuda.benchmark import (
    compare_mode_summaries,
    validate_comparable_contracts,
)
from vllm_latchmoe_cuda.correctness import (
    CorrectnessMismatchError,
    compare_exact_token_results,
    require_exact_token_gate,
)


def _contract(**overrides):
    value = {
        "selection_strategy": "midpoint_stratified_v1",
        "eligible_layer_ids": list(range(48)),
        "selected_layer_ids": [2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46],
        "plan_id": "a" * 64,
        "parameter_names": ["model.layers.2.mlp.experts.w13_weight"],
        "host_bytes": 123,
        "resident_weight_bytes": 456,
        "backend_hbm_cache_bytes": 64,
        "kv_reserve_bytes": 512,
        "graph_policy": "piecewise",
        "workload_contract_sha256": "b" * 64,
        "source_identity": "c" * 64,
    }
    value.update(overrides)
    return value


def _result(ids):
    return {
        "outputs": [
            {"prompt": "p", "prompt_token_ids": [1], "token_ids": ids}
        ]
    }


def test_comparable_contract_checks_all_identity_fields():
    left = _contract()
    right = _contract()
    assert validate_comparable_contracts(left, right)["reservation_equal"]
    right["plan_id"] = "d" * 64
    with pytest.raises(ValueError, match="plan_id"):
        validate_comparable_contracts(left, right)


def test_comparable_contract_requires_equal_backend_hbm_cache():
    left = _contract(backend_hbm_cache_bytes=64)
    right = _contract(backend_hbm_cache_bytes=32)

    with pytest.raises(ValueError, match="backend_hbm_cache_bytes"):
        validate_comparable_contracts(left, right)


def test_exact_token_gate_requires_all_three_runs():
    native = _result([1, 2])
    uva = _result([1, 2])
    latchmoe = _result([1, 3])
    report = compare_exact_token_results(native, uva, latchmoe)
    assert report["match"] is False
    with pytest.raises(CorrectnessMismatchError, match="latchmoe"):
        require_exact_token_gate(native, uva, latchmoe)


def test_exact_token_gate_accepts_identical_outputs():
    result = _result([1, 2])
    assert require_exact_token_gate(result, result, result)["match"] is True


def test_comparison_rejects_legacy_evidence():
    common = {
        "workload_contract_sha256": "b" * 64,
        "offload_telemetry": {"actual_offload_bytes": 123},
        "comparison_contract": {**_contract(), "legacy_evidence": True},
        "metrics": {},
    }
    with pytest.raises(ValueError, match="legacy temporary-bank"):
        compare_mode_summaries(
            {**common, "mode": "uva"},
            {**common, "mode": "latchmoe-eager"},
        )


def test_runtime_ledger_enforces_zero_temporary_bank_and_pair_math():
    ledger = {
        "temporary_bank_bytes": 0,
        "num_tokens": 4,
        "top_k": 2,
        "pair_count": 8,
        "selected_layer_forward_count": 3,
        "combine_count": 3,
    }
    report = validate_runtime_ledger(ledger)
    assert report["pair_count"] == 8
    ledger["pair_count"] = 7
    with pytest.raises(BudgetContractError, match="pair_count"):
        validate_runtime_ledger(ledger)


def test_runtime_ledger_rejects_temporary_bank_and_wrong_dynamic_bytes():
    with pytest.raises(BudgetContractError, match="temporary_bank_bytes"):
        validate_runtime_ledger({"temporary_bank_bytes": 1})
    with pytest.raises(BudgetContractError, match="allocated_dynamic"):
        validate_runtime_ledger(
            {
                "temporary_bank_bytes": 0,
                "allocated_dynamic_weight_bytes": 9,
            },
            selected_layer_count=2,
            slots_per_layer=2,
            bytes_per_expert=4,
        )


def test_profile_requires_one_unified_layout_per_main_cache_forward():
    event = {
        "event": "main_cache_waves",
        "pair_layout": "unified_token_expert_v1",
        "pair_layout_build_count": 1,
    }

    assert validate_profile_events([event])["unified_pair_layouts"] == 1
    event["pair_layout"] = "per_wave_mask"
    with pytest.raises(BudgetContractError, match="unified pair layout"):
        validate_profile_events([event])
