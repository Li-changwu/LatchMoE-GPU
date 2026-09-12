"""Deterministic, model-adaptive CUDA residency planning.

This module deliberately has no CUDA or vLLM imports.  The parent process can
therefore create a plan before workers (and before importing vLLM) and pass the
same signed-by-content document to every worker.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import PlanValidationError, ResidencyBudgetError

GIB = 1 << 30
PLANNER_VERSION = "2026-09-main-cache-v1"


def _canonical(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    return value


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_plain(v) for v in value]
    return value


@dataclass(frozen=True)
class LayerExpertBytes:
    layer_id: int
    routed_expert_bytes: int
    shared_expert_bytes: int = 0

    def __post_init__(self) -> None:
        if self.layer_id < 0:
            raise PlanValidationError("layer_id must be non-negative")
        if self.routed_expert_bytes <= 0:
            raise PlanValidationError("routed_expert_bytes must be positive")
        if self.shared_expert_bytes < 0:
            raise PlanValidationError("shared_expert_bytes must be non-negative")

    def to_jsonable(self) -> dict[str, int]:
        return {
            "layer_id": self.layer_id,
            "routed_expert_bytes": self.routed_expert_bytes,
            "shared_expert_bytes": self.shared_expert_bytes,
        }


@dataclass(frozen=True)
class CudaResidencyPlan:
    schema_version: int
    planner_version: str
    selection_strategy: str
    plan_id: str
    model_fingerprint: Mapping[str, object]
    num_experts: int
    requested_offload_bytes: int
    effective_offloaded_bytes: int
    eligible_layer_ids: tuple[int, ...]
    offloaded_layer_ids: tuple[int, ...]
    resident_layer_ids: tuple[int, ...]
    layer_expert_bytes: tuple[LayerExpertBytes, ...]
    recommended_num_slots: int
    effective_num_slots: int
    main_slot_cache_bytes: int
    net_hbm_saved_bytes: int
    ledger: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "model_fingerprint", _freeze(self.model_fingerprint))
        object.__setattr__(self, "ledger", _freeze(self.ledger))
        for name in (
            "eligible_layer_ids",
            "offloaded_layer_ids",
            "resident_layer_ids",
        ):
            values = tuple(int(v) for v in getattr(self, name))
            if values != tuple(dict.fromkeys(values)) or values != tuple(sorted(values)):
                raise PlanValidationError(f"{name} must be sorted and unique")
            object.__setattr__(self, name, values)
        if tuple(sorted(self.offloaded_layer_ids + self.resident_layer_ids)) != (
            self.eligible_layer_ids
        ):
            raise PlanValidationError(
                "offloaded and resident layers must partition eligible layers"
            )
        if self.num_experts <= 0 or self.effective_num_slots <= 0:
            raise PlanValidationError("expert and slot counts must be positive")
        if self.effective_num_slots > self.num_experts:
            raise PlanValidationError("effective_num_slots exceeds num_experts")
        if self.requested_offload_bytes < 0 or self.effective_offloaded_bytes < 0:
            raise PlanValidationError("offload bytes must be non-negative")

    @property
    def offloaded_layer_count(self) -> int:
        return len(self.offloaded_layer_ids)

    def to_jsonable(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "planner_version": self.planner_version,
            "selection_strategy": self.selection_strategy,
            "plan_id": self.plan_id,
            "model_fingerprint": _plain(self.model_fingerprint),
            "num_experts": self.num_experts,
            "requested_offload_bytes": self.requested_offload_bytes,
            "effective_offloaded_bytes": self.effective_offloaded_bytes,
            "eligible_layer_ids": list(self.eligible_layer_ids),
            "offloaded_layer_ids": list(self.offloaded_layer_ids),
            "resident_layer_ids": list(self.resident_layer_ids),
            "layer_expert_bytes": [
                value.to_jsonable() for value in self.layer_expert_bytes
            ],
            "recommended_num_slots": self.recommended_num_slots,
            "effective_num_slots": self.effective_num_slots,
            "main_slot_cache_bytes": self.main_slot_cache_bytes,
            "net_hbm_saved_bytes": self.net_hbm_saved_bytes,
            "ledger": _plain(self.ledger),
        }


def _model_type(config: Mapping[str, object]) -> str:
    value = config.get("model_type")
    if not isinstance(value, str) or not value:
        raise PlanValidationError("model_config.model_type is required")
    return value.lower()


def _num_experts(config: Mapping[str, object]) -> int:
    value = config.get("num_experts", config.get("n_routed_experts"))
    if value is None:
        raise PlanValidationError("model_config.num_experts is required")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("num_experts must be an integer") from exc
    if result <= 0:
        raise PlanValidationError("num_experts must be positive")
    return result


def _eligible_ids(config: Mapping[str, object]) -> tuple[int, ...]:
    dense = {int(v) for v in config.get("dense_layer_ids", ())}
    explicit = config.get("moe_layer_ids")
    if explicit is None:
        count = config.get("num_hidden_layers")
        if count is None:
            raise PlanValidationError(
                "model_config must provide moe_layer_ids or num_hidden_layers"
            )
        try:
            explicit_ids = range(int(count))
        except (TypeError, ValueError) as exc:
            raise PlanValidationError("num_hidden_layers must be an integer") from exc
    else:
        if not isinstance(explicit, (list, tuple)):
            raise PlanValidationError("moe_layer_ids must be a sequence")
        normalized = tuple(int(v) for v in explicit)
        if normalized != tuple(sorted(normalized)) or len(normalized) != len(set(normalized)):
            raise PlanValidationError("moe_layer_ids must be sorted and unique")
        explicit_ids = normalized
    ids = tuple(int(v) for v in explicit_ids if int(v) not in dense)
    if any(v < 0 for v in ids):
        raise PlanValidationError("layer ids must be non-negative")
    if not ids:
        raise PlanValidationError("model has no eligible routed-MoE layers")
    return ids


def _dtype_bytes(config: Mapping[str, object]) -> int:
    dtype = str(config.get("torch_dtype", config.get("dtype", ""))).lower()
    if dtype.startswith("torch."):
        dtype = dtype.removeprefix("torch.")
    if dtype not in {"bfloat16", "bf16"}:
        raise PlanValidationError("only BF16 residency planning is supported")
    return 2


def _derived_layer_bytes(config: Mapping[str, object], experts: int) -> int:
    model_type = _model_type(config)
    if model_type not in {"qwen3_moe", "glm4_moe_lite"}:
        raise PlanValidationError(
            "routed expert byte metadata is required for this model architecture"
        )
    try:
        hidden = int(config["hidden_size"])
        intermediate = int(config["moe_intermediate_size"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanValidationError(
            "supported BF16 MoE planning requires hidden_size and moe_intermediate_size"
        ) from exc
    if hidden <= 0 or intermediate <= 0:
        raise PlanValidationError("hidden_size and moe_intermediate_size must be positive")
    return 3 * hidden * intermediate * experts * _dtype_bytes(config)


def _metadata_table(
    config: Mapping[str, object],
    eligible: tuple[int, ...],
    experts: int,
    metadata: Sequence[Mapping[str, object]] | None,
) -> tuple[LayerExpertBytes, ...]:
    if metadata is None:
        value = _derived_layer_bytes(config, experts)
        return tuple(LayerExpertBytes(layer_id, value) for layer_id in eligible)
    by_id: dict[int, LayerExpertBytes] = {}
    for item in metadata:
        try:
            layer_id = int(item["layer_id"])
            routed = int(item["routed_expert_bytes"])
            shared = int(item.get("shared_expert_bytes", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise PlanValidationError("invalid layer_metadata entry") from exc
        if layer_id in by_id:
            raise PlanValidationError(f"duplicate layer metadata for layer {layer_id}")
        by_id[layer_id] = LayerExpertBytes(layer_id, routed, shared)
    if set(by_id) != set(eligible):
        raise PlanValidationError(
            "layer_metadata must contain exactly the ordered eligible layers"
        )
    return tuple(by_id[layer_id] for layer_id in eligible)


def _midpoint_indices(count: int, selected: int) -> tuple[int, ...]:
    if selected == 0:
        return ()
    used: set[int] = set()
    values: list[int] = []
    for j in range(selected):
        index = min(math.floor((j + 0.5) * count / selected), count - 1)
        if index in used:
            found = None
            for distance in range(1, count):
                for candidate in (index - distance, index + distance):
                    if 0 <= candidate < count and candidate not in used:
                        found = candidate
                        break
                if found is not None:
                    break
            if found is None:
                raise PlanValidationError("could not produce unique midpoint selection")
            index = found
        used.add(index)
        values.append(index)
    return tuple(sorted(values))


def _profile_bytes_and_scores(path: str) -> tuple[str, dict[int, float]]:
    profile = Path(path)
    raw = profile.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    try:
        payload: Any = json.loads(raw)
    except json.JSONDecodeError:
        payload = []
        for line in raw.splitlines():
            if line.strip():
                try:
                    payload.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    scores: dict[int, float] = {}

    def consume(value: Any) -> None:
        if isinstance(value, Mapping):
            for key in ("layer_scores", "scores", "hot_layers", "layer_heat"):
                table = value.get(key)
                if isinstance(table, Mapping):
                    for layer, score in table.items():
                        try:
                            scores[int(layer)] = float(score)
                        except (TypeError, ValueError):
                            pass
                elif isinstance(table, list):
                    for entry in table:
                        if isinstance(entry, Mapping) and "layer_id" in entry:
                            try:
                                scores[int(entry["layer_id"])] = float(
                                    entry.get("score", entry.get("count", 0))
                                )
                            except (TypeError, ValueError):
                                pass
            if "layer_id" in value and any(
                key in value for key in ("score", "count", "hits")
            ):
                try:
                    scores[int(value["layer_id"])] = float(
                        value.get("score", value.get("count", value.get("hits", 0)))
                    )
                except (TypeError, ValueError):
                    pass
            for child in value.values():
                if isinstance(child, (Mapping, list, tuple)):
                    consume(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                consume(child)

    consume(payload)
    return digest, scores


def _plan_payload_without_id(
    *,
    schema_version: int,
    planner_version: str,
    selection_strategy: str,
    model_fingerprint: Mapping[str, object],
    num_experts: int,
    requested_offload_bytes: int,
    effective_offloaded_bytes: int,
    eligible_layer_ids: tuple[int, ...],
    offloaded_layer_ids: tuple[int, ...],
    resident_layer_ids: tuple[int, ...],
    layer_expert_bytes: tuple[LayerExpertBytes, ...],
    recommended_num_slots: int,
    effective_num_slots: int,
    main_slot_cache_bytes: int,
    net_hbm_saved_bytes: int,
    ledger: Mapping[str, object],
) -> dict[str, object]:
    return {
        "schema_version": schema_version,
        "planner_version": planner_version,
        "selection_strategy": selection_strategy,
        "model_fingerprint": _plain(model_fingerprint),
        "num_experts": num_experts,
        "requested_offload_bytes": requested_offload_bytes,
        "effective_offloaded_bytes": effective_offloaded_bytes,
        "eligible_layer_ids": list(eligible_layer_ids),
        "offloaded_layer_ids": list(offloaded_layer_ids),
        "resident_layer_ids": list(resident_layer_ids),
        "layer_expert_bytes": [v.to_jsonable() for v in layer_expert_bytes],
        "recommended_num_slots": recommended_num_slots,
        "effective_num_slots": effective_num_slots,
        "main_slot_cache_bytes": main_slot_cache_bytes,
        "net_hbm_saved_bytes": net_hbm_saved_bytes,
        "ledger": _plain(ledger),
    }


def _make_plan(
    *,
    strategy: str,
    base_strategy: str,
    fingerprint: Mapping[str, object],
    experts: int,
    requested: int,
    eligible: tuple[int, ...],
    selected: tuple[int, ...],
    table: tuple[LayerExpertBytes, ...],
    slots: int,
    ledger: Mapping[str, object],
) -> CudaResidencyPlan:
    by_id = {item.layer_id: item for item in table}
    effective = sum(by_id[layer].routed_expert_bytes for layer in selected)
    per_expert_cache = sum(
        (slots * by_id[layer].routed_expert_bytes + experts - 1) // experts
        for layer in selected
    )
    resident = tuple(layer for layer in eligible if layer not in set(selected))
    full_ledger = dict(ledger)
    full_ledger.setdefault("base_selection_strategy", base_strategy)
    full_ledger.setdefault("selected_layer_count", len(selected))
    payload = _plan_payload_without_id(
        schema_version=2,
        planner_version=PLANNER_VERSION,
        selection_strategy=strategy,
        model_fingerprint=fingerprint,
        num_experts=experts,
        requested_offload_bytes=requested,
        effective_offloaded_bytes=effective,
        eligible_layer_ids=eligible,
        offloaded_layer_ids=selected,
        resident_layer_ids=resident,
        layer_expert_bytes=table,
        recommended_num_slots=slots,
        effective_num_slots=slots,
        main_slot_cache_bytes=per_expert_cache,
        net_hbm_saved_bytes=effective - per_expert_cache,
        ledger=full_ledger,
    )
    plan_id = hashlib.sha256(_canonical(payload)).hexdigest()
    return CudaResidencyPlan(
        schema_version=2,
        planner_version=PLANNER_VERSION,
        selection_strategy=strategy,
        plan_id=plan_id,
        model_fingerprint=fingerprint,
        num_experts=experts,
        requested_offload_bytes=requested,
        effective_offloaded_bytes=effective,
        eligible_layer_ids=eligible,
        offloaded_layer_ids=selected,
        resident_layer_ids=resident,
        layer_expert_bytes=table,
        recommended_num_slots=slots,
        effective_num_slots=slots,
        main_slot_cache_bytes=per_expert_cache,
        net_hbm_saved_bytes=effective - per_expert_cache,
        ledger=full_ledger,
    )


def build_residency_plan(
    requested_offload_gib: float,
    model_config: Mapping[str, object],
    *,
    max_capture_size: int,
    top_k: int,
    device_total_bytes: int,
    kv_reserve_bytes: int,
    layer_metadata: Sequence[Mapping[str, object]] | None = None,
    profile_path: str | None = None,
) -> CudaResidencyPlan:
    try:
        requested_float = float(requested_offload_gib)
    except (TypeError, ValueError) as exc:
        raise PlanValidationError("requested_offload_gib must be numeric") from exc
    if not math.isfinite(requested_float) or requested_float < 0:
        raise PlanValidationError("requested_offload_gib must be finite and non-negative")
    if max_capture_size <= 0 or top_k <= 0:
        raise PlanValidationError("max_capture_size and top_k must be positive")
    if device_total_bytes < 0 or kv_reserve_bytes < 0:
        raise PlanValidationError("device and KV bytes must be non-negative")
    requested = math.ceil(requested_float * GIB)
    experts = _num_experts(model_config)
    eligible = _eligible_ids(model_config)
    table = _metadata_table(model_config, eligible, experts, layer_metadata)
    values = tuple(item.routed_expert_bytes for item in table)
    uniform = len(set(values)) == 1
    if uniform:
        layer_bytes = values[0]
        count = min(len(eligible), math.ceil(requested / layer_bytes)) if requested else 0
        selected = tuple(eligible[index] for index in _midpoint_indices(len(eligible), count))
        strategy = "midpoint_stratified_v1"
    else:
        accumulated = 0
        selected_values: list[int] = []
        if requested:
            for item in table:
                if accumulated >= requested:
                    break
                selected_values.append(item.layer_id)
                accumulated += item.routed_expert_bytes
        selected = tuple(selected_values)
        strategy = "ordered_prefix_bytes_v1"
    capped = bool(requested and len(selected) == len(eligible) and requested > sum(values))
    slots = min(experts, int(max_capture_size) * int(top_k))
    fingerprint = {
        "model_type": _model_type(model_config),
        "torch_dtype": str(model_config.get("torch_dtype", model_config.get("dtype", ""))),
        "num_hidden_layers": model_config.get("num_hidden_layers"),
        "moe_layer_ids": list(model_config.get("moe_layer_ids", []))
        if model_config.get("moe_layer_ids") is not None
        else None,
        "dense_layer_ids": list(model_config.get("dense_layer_ids", [])),
        "hidden_size": model_config.get("hidden_size"),
        "moe_intermediate_size": model_config.get("moe_intermediate_size"),
        "num_experts": experts,
    }
    # Preserve content identity fields supplied by the model loader.  They are
    # intentionally part of the plan digest when present, but are not guessed
    # from a model name.
    for key in (
        "model_path",
        "revision",
        "config_sha256",
        "weight_index_sha256",
    ):
        if key in model_config:
            fingerprint[key] = model_config[key]
    ledger: dict[str, object] = {
        "capped_to_eligible_layers": capped,
        "requested_offload_gib": requested_float,
        "eligible_layer_count": len(eligible),
        "uniform_layer_bytes": uniform,
        "minimum_slots": slots,
        "kv_reserve_bytes": int(kv_reserve_bytes),
        "resident_shared_weight_bytes": sum(
            item.shared_expert_bytes for item in table
        ),
    }
    if profile_path is not None:
        profile_hash, scores = _profile_bytes_and_scores(profile_path)
        base_selected = selected
        profile_swap_valid = False
        if scores:
            ranked = tuple(
                sorted(eligible, key=lambda layer: (-scores.get(layer, float("-inf")), layer))
            )
            candidate = tuple(sorted(ranked[: len(base_selected)]))
            by_id = {item.layer_id: item for item in table}
            if sum(by_id[layer].routed_expert_bytes for layer in candidate) == sum(
                by_id[layer].routed_expert_bytes for layer in base_selected
            ):
                selected = candidate
                profile_swap_valid = True
        ledger.update(
            profile_sha256=profile_hash,
            profile_path=str(profile_path),
            profile_score_count=len(scores),
            profile_base_selected_layer_ids=list(base_selected),
        )
        strategy = "profile_guided_swap_v1" if profile_swap_valid else strategy
    total = sum(values)
    # Keep the HBM check in terms of actual routed expert bytes, never shared bytes.
    by_id = {item.layer_id: item for item in table}
    selected_bytes = sum(by_id[layer].routed_expert_bytes for layer in selected)
    cache_bytes = sum(
        (slots * by_id[layer].routed_expert_bytes + experts - 1) // experts
        for layer in selected
    )
    net_saved = selected_bytes - cache_bytes
    ledger["host_routed_expert_bytes"] = int(selected_bytes)
    ledger["dynamic_slot_bytes"] = int(cache_bytes)
    estimated_hbm = total - selected_bytes + cache_bytes + int(kv_reserve_bytes)
    if selected and net_saved <= 0:
        raise ResidencyBudgetError(
            "minimum slot cache does not save HBM; refusing shared-pool fallback"
        )
    if device_total_bytes and estimated_hbm > int(device_total_bytes):
        raise ResidencyBudgetError(
            f"estimated HBM {estimated_hbm} exceeds device/KV boundary {device_total_bytes}"
        )
    plan = _make_plan(
        strategy=strategy,
        base_strategy=(
            "midpoint_stratified_v1" if uniform else "ordered_prefix_bytes_v1"
        ),
        fingerprint=fingerprint,
        experts=experts,
        requested=requested,
        eligible=eligible,
        selected=selected,
        table=table,
        slots=slots,
        ledger={**ledger, "effective_hbm_bytes": estimated_hbm},
    )
    if plan.effective_offloaded_bytes != selected_bytes:
        raise PlanValidationError("internal effective byte calculation mismatch")
    return plan


def serialize_residency_plan(plan: CudaResidencyPlan) -> str:
    return json.dumps(plan.to_jsonable(), sort_keys=True, separators=(",", ":"))


def deserialize_residency_plan(raw: str) -> CudaResidencyPlan:
    try:
        payload = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as exc:
        raise PlanValidationError("invalid residency plan JSON") from exc
    if not isinstance(payload, Mapping):
        raise PlanValidationError("residency plan must be a JSON object")
    try:
        table = tuple(
            LayerExpertBytes(
                int(item["layer_id"]),
                int(item["routed_expert_bytes"]),
                int(item.get("shared_expert_bytes", 0)),
            )
            for item in payload["layer_expert_bytes"]
        )
        plan = CudaResidencyPlan(
            schema_version=int(payload["schema_version"]),
            planner_version=str(payload["planner_version"]),
            selection_strategy=str(payload["selection_strategy"]),
            plan_id=str(payload["plan_id"]),
            model_fingerprint=payload["model_fingerprint"],
            num_experts=int(payload["num_experts"]),
            requested_offload_bytes=int(payload["requested_offload_bytes"]),
            effective_offloaded_bytes=int(payload["effective_offloaded_bytes"]),
            eligible_layer_ids=tuple(int(v) for v in payload["eligible_layer_ids"]),
            offloaded_layer_ids=tuple(int(v) for v in payload["offloaded_layer_ids"]),
            resident_layer_ids=tuple(int(v) for v in payload["resident_layer_ids"]),
            layer_expert_bytes=table,
            recommended_num_slots=int(payload["recommended_num_slots"]),
            effective_num_slots=int(payload["effective_num_slots"]),
            main_slot_cache_bytes=int(payload["main_slot_cache_bytes"]),
            net_hbm_saved_bytes=int(payload["net_hbm_saved_bytes"]),
            ledger=payload["ledger"],
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise PlanValidationError("invalid residency plan fields") from exc
    expected_payload = plan.to_jsonable()
    expected_payload.pop("plan_id", None)
    expected = hashlib.sha256(_canonical(expected_payload)).hexdigest()
    if plan.schema_version != 2:
        raise PlanValidationError(f"unsupported residency plan schema={plan.schema_version}")
    if plan.plan_id != expected:
        raise PlanValidationError(
            f"residency plan id mismatch: expected={expected}, actual={plan.plan_id}"
        )
    return plan


def validate_plan_against_model(
    plan: CudaResidencyPlan, model_config: Mapping[str, object]
) -> None:
    experts = _num_experts(model_config)
    eligible = _eligible_ids(model_config)
    if experts != plan.num_experts:
        raise PlanValidationError("plan num_experts does not match model")
    if eligible != plan.eligible_layer_ids:
        raise PlanValidationError("plan eligible layer IDs do not match model")
    if tuple(item.layer_id for item in plan.layer_expert_bytes) != eligible:
        raise PlanValidationError("plan layer byte table does not match model")
    dtype = str(model_config.get("torch_dtype", model_config.get("dtype", ""))).lower()
    expected_type = _model_type(model_config)
    if plan.model_fingerprint.get("model_type") != expected_type:
        raise PlanValidationError("plan model fingerprint does not match model")
    if dtype and str(plan.model_fingerprint.get("torch_dtype", "")).lower() != dtype:
        raise PlanValidationError("plan dtype fingerprint does not match model")
