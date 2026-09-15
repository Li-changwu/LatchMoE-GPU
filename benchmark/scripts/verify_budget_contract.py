#!/usr/bin/env python3
"""Fail-closed verification for comparable LatchMoE benchmark evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from vllm_latchmoe_cuda.benchmark import (
    COMPARISON_CONTRACT_FIELDS,
    validate_comparable_contracts,
)


class BudgetContractError(ValueError):
    """Raised when benchmark evidence cannot support a fair comparison."""


def _as_ids(value: Any) -> tuple[int, ...]:
    if not isinstance(value, (list, tuple)):
        raise BudgetContractError("layer IDs must be an array")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise BudgetContractError("layer IDs must be integers") from exc
    if result != tuple(sorted(set(result))):
        raise BudgetContractError("layer IDs must be sorted and unique")
    return result


def _value(document: Mapping[str, Any], field: str) -> Any:
    nested = document.get("comparison_contract")
    if isinstance(nested, Mapping) and field in nested:
        return nested[field]
    nested = document.get("contract")
    if isinstance(nested, Mapping) and field in nested:
        return nested[field]
    return document.get(field)


def validate_plan_evidence(document: Mapping[str, Any]) -> dict[str, Any]:
    """Validate placement identity and reject legacy placement as production."""
    strategy = _value(document, "selection_strategy")
    if not isinstance(strategy, str) or not strategy:
        raise BudgetContractError("selection_strategy is required")
    eligible = _as_ids(_value(document, "eligible_layer_ids"))
    selected = _as_ids(_value(document, "selected_layer_ids"))
    if not set(selected).issubset(eligible):
        raise BudgetContractError("selected layer IDs are not eligible")
    if _value(document, "legacy_evidence") is True or strategy.startswith("legacy"):
        raise BudgetContractError(
            "legacy temporary-bank/shared-pool evidence cannot be promoted"
        )
    hbm_cache_bytes = _value(document, "backend_hbm_cache_bytes")
    if (
        not isinstance(hbm_cache_bytes, int)
        or isinstance(hbm_cache_bytes, bool)
        or hbm_cache_bytes < 0
    ):
        raise BudgetContractError(
            "backend_hbm_cache_bytes must be a non-negative integer"
        )
    model_value = document.get("model", "")
    model_type = document.get("model_type")
    if model_type is None and isinstance(model_value, Mapping):
        model_type = model_value.get("model_type", model_value.get("architectures", ""))
    model_type = str(model_type or model_value).lower()
    if "qwen3" in model_type and len(eligible) == 48 and len(selected) == 12:
        expected = (2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46)
        if strategy != "midpoint_stratified_v1" or selected != expected:
            raise BudgetContractError("Qwen3 E=48,K=12 placement is not midpoint stratified")
    dense = set(int(item) for item in document.get("dense_layer_ids", ()))
    non_moe = set(int(item) for item in document.get("non_moe_layer_ids", ()))
    if dense.intersection(eligible) or non_moe.intersection(eligible):
        raise BudgetContractError("eligible layers contain dense/non-MoE layers")
    return {
        "selection_strategy": strategy,
        "eligible_layer_ids": list(eligible),
        "selected_layer_ids": list(selected),
        "plan_id": _value(document, "plan_id"),
        "backend_hbm_cache_bytes": hbm_cache_bytes,
    }


def _events(profile: Iterable[Mapping[str, Any]] | Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if isinstance(profile, Mapping):
        if isinstance(profile.get("events"), list):
            profile = profile["events"]
        elif any(key in profile for key in ("event", "pair_count", "combine_count", "temporary_bank_bytes")):
            profile = (profile,)
        else:
            profile = ()
    return [event for event in profile if isinstance(event, Mapping)]


def validate_profile_events(
    profile: Iterable[Mapping[str, Any]] | Mapping[str, Any],
    *,
    capacity: int | None = None,
) -> dict[str, Any]:
    """Reject unsupported transfer/overlap claims in runtime telemetry."""
    events = _events(profile)
    errors: list[str] = []
    temporary = [e for e in events if e.get("event") == "temporary_bank" or e.get("temporary_bank_bytes", 0)]
    if temporary:
        errors.append("temporary bank evidence is forbidden")
    for event in events:
        name = str(event.get("event", ""))
        h2d = int(event.get("h2d_bytes", event.get("host_to_device_bytes", 0)) or 0)
        mode = str(event.get("stage_mode", ""))
        if name in {"resident_layer", "resident", "direct_slots"} and h2d:
            errors.append("resident layer has H2D traffic")
        if name in {"hit_wave", "exact_wave", "main_cache_wave"} and event.get("hit_wave") and h2d:
            errors.append("hit wave has H2D traffic")
        if mode == "temporary_bank":
            errors.append("temporary bank stage mode is forbidden")
        if "pair_count" in event and "expected_pair_count" in event:
            if int(event["pair_count"]) != int(event["expected_pair_count"]):
                errors.append("pair count is incomplete")
        if name in {"combine", "native_combine"} and int(event.get("combine_count", 1)) != 1:
            errors.append("combine count must be one per layer")
        if name == "main_cache_waves" and (
            event.get("pair_layout") != "unified_token_expert_v1"
            or int(event.get("pair_layout_build_count", 0)) != 1
        ):
            errors.append("Main Cache forward must build one unified pair layout")
        candidate = bool(event.get("overlap_candidate"))
        actual = bool(event.get("actual_overlap"))
        if candidate and actual:
            starts = [event.get(key) for key in ("h2d_start_ms", "compute_start_ms")]
            ends = [event.get(key) for key in ("h2d_end_ms", "compute_end_ms")]
            if any(value is None for value in starts + ends):
                errors.append("overlap candidate has no actual event window")
            elif max(starts) >= min(ends):
                errors.append("overlap event window is empty")
        if actual and event.get("hit_count") is not None and int(event.get("hit_count", 0)) == 0:
            errors.append("overlap claim has no hit wave")
        h = event.get("h")
        c = event.get("C", event.get("capacity", capacity))
        p = event.get("P", event.get("pair_count"))
        if actual and h is not None and c is not None and p is not None:
            if int(h) == 0 or int(h) >= int(c) or int(c) >= int(p):
                errors.append("overlap claimed for serial boundary")
    if errors:
        raise BudgetContractError("; ".join(dict.fromkeys(errors)))
    return {
        "events": len(events),
        "temporary_bank_bytes": sum(int(e.get("temporary_bank_bytes", 0) or 0) for e in events),
        "overlap_candidates": sum(bool(e.get("overlap_candidate")) for e in events),
        "actual_overlaps": sum(bool(e.get("actual_overlap")) for e in events),
        "unified_pair_layouts": sum(
            e.get("event") == "main_cache_waves"
            and e.get("pair_layout") == "unified_token_expert_v1"
            for e in events
        ),
    }


def validate_runtime_ledger(
    ledger: Mapping[str, Any],
    *,
    selected_layer_count: int | None = None,
    slots_per_layer: int | None = None,
    bytes_per_expert: int | None = None,
) -> dict[str, Any]:
    """Validate the resource counters required by the 12-layer gate."""
    temporary = int(ledger.get("temporary_bank_bytes", 0) or 0)
    if temporary != 0:
        raise BudgetContractError("temporary_bank_bytes must be zero")
    if "pair_count" in ledger and "num_tokens" in ledger and "top_k" in ledger:
        expected = int(ledger["num_tokens"]) * int(ledger["top_k"])
        if int(ledger["pair_count"]) != expected:
            raise BudgetContractError(
                f"pair_count must equal num_tokens*top_k ({expected})"
            )
    if "combine_count" in ledger and "selected_layer_forward_count" in ledger:
        if int(ledger["combine_count"]) != int(ledger["selected_layer_forward_count"]):
            raise BudgetContractError("combine_count must equal selected layer forward count")
    if (
        slots_per_layer is not None
        and bytes_per_expert is not None
        and selected_layer_count is not None
        and "allocated_dynamic_weight_bytes" in ledger
    ):
        expected = int(selected_layer_count) * int(slots_per_layer) * int(bytes_per_expert)
        if int(ledger["allocated_dynamic_weight_bytes"]) != expected:
            raise BudgetContractError(
                "allocated_dynamic_weight_bytes does not match selected layers*slots*bytes_per_expert"
            )
    return {
        "temporary_bank_bytes": temporary,
        "pair_count": ledger.get("pair_count"),
        "combine_count": ledger.get("combine_count"),
        "allocated_dynamic_weight_bytes": ledger.get("allocated_dynamic_weight_bytes"),
    }


def verify_exact_token_gate(
    native: Mapping[str, Any],
    uva: Mapping[str, Any],
    latchmoe: Mapping[str, Any],
) -> dict[str, Any]:
    """Require exact token IDs for native, exact UVA, and LatchMoE outputs."""
    outputs = []
    prompts = native.get("prompts")
    sampling = native.get("sampling")
    for name, document in (("native", native), ("uva", uva), ("latchmoe", latchmoe)):
        if name != "native":
            if document.get("prompts") != prompts:
                raise BudgetContractError(f"{name} prompt contract differs")
            if document.get("sampling") != sampling:
                raise BudgetContractError(f"{name} sampling contract differs")
        value = document.get("outputs")
        if not isinstance(value, list) or not value:
            raise BudgetContractError(f"{name} output list is missing")
        outputs.append((name, value))
    reference = outputs[0][1]
    for name, candidate in outputs[1:]:
        if len(candidate) != len(reference):
            raise BudgetContractError(f"{name} request count differs")
        for index, (expected, actual) in enumerate(zip(reference, candidate)):
            if expected.get("prompt") != actual.get("prompt"):
                raise BudgetContractError(f"{name} prompt differs at request {index}")
            if expected.get("prompt_token_ids") != actual.get("prompt_token_ids"):
                raise BudgetContractError(f"{name} prompt token IDs differ at request {index}")
            if expected.get("token_ids") != actual.get("token_ids"):
                raise BudgetContractError(f"{name} token IDs differ at request {index}")
    return {"request_count": len(reference), "exact_match": True}


def verify_budget_contract(
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    profile: Iterable[Mapping[str, Any]] | Mapping[str, Any] | None = None,
    ledger: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_plan_evidence(baseline)
    validate_plan_evidence(candidate)
    comparison = validate_comparable_contracts(dict(baseline), dict(candidate))
    profile_report = validate_profile_events(profile) if profile is not None else None
    ledger_report = validate_runtime_ledger(ledger) if ledger is not None else None
    return {
        "pass": True,
        "comparison": comparison,
        "profile": profile_report,
        "ledger": ledger_report,
    }


def _load(path: Path) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        return {
            "events": [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        }
    if path.is_dir():
        candidates = (
            "summary.json",
            "result.json",
            "correctness.json",
            "outputs.json",
        )
        for name in candidates:
            candidate = path / name
            if candidate.is_file():
                path = candidate
                break
        else:
            discovered = sorted(path.glob("**/*.json"))
            if not discovered:
                raise BudgetContractError(f"no JSON result found under {path}")
            path = discovered[0]
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise BudgetContractError(f"{path} must contain a JSON object")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify LatchMoE budget and evidence contract")
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--native", type=Path)
    parser.add_argument("--uva", type=Path)
    parser.add_argument("--latchmoe", type=Path)
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--native-json", type=Path)
    parser.add_argument("--uva-json", type=Path)
    parser.add_argument("--latchmoe-json", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.baseline is None or args.candidate is None:
            if not (args.uva and args.latchmoe):
                raise BudgetContractError("provide --baseline/--candidate or --uva/--latchmoe")
            baseline_path, candidate_path = args.uva, args.latchmoe
        else:
            baseline_path, candidate_path = args.baseline, args.candidate
        profile = _load(args.profile) if args.profile else None
        ledger = _load(args.ledger) if args.ledger else None
        if profile is None and args.latchmoe and args.latchmoe.is_dir():
            profile_path = args.latchmoe / "profile.jsonl"
            if profile_path.is_file():
                profile = {"events": [json.loads(line) for line in profile_path.read_text().splitlines() if line]}
        report = verify_budget_contract(
            _load(baseline_path), _load(candidate_path), profile=profile, ledger=ledger
        )
        native_path = args.native_json or args.native
        uva_path = args.uva_json or args.uva
        latchmoe_path = args.latchmoe_json or args.latchmoe
        if native_path and uva_path and latchmoe_path:
            report["exact_token_gate"] = verify_exact_token_gate(
                _load(native_path), _load(uva_path), _load(latchmoe_path)
            )
    except (OSError, json.JSONDecodeError, BudgetContractError, ValueError) as exc:
        report = {"pass": False, "error": str(exc)}
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"FAIL: {exc}")
        return 1
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("PASS: budget contract and evidence gate")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
