from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Sequence

from .correctness import STOCK_UVA_CPU_OFFLOAD_GB
from .manifest import OffloadManifest, canonical_json_bytes


BENCHMARK_MODES = (
    "uva",
    "uva-piecewise",
    "uva-full-and-piecewise",
    "latchmoe-eager",
    "latchmoe-async-eager",
    "latchmoe-piecewise",
)
SUMMARY_METRICS = (
    "median_ttft_ms",
    "mean_ttft_ms",
    "p99_ttft_ms",
    "median_tpot_ms",
    "mean_tpot_ms",
    "p99_tpot_ms",
    "output_throughput",
    "request_throughput",
)

# Fields that define whether two measurements exercised the same memory and
# execution contract.  Keeping this list in the library lets both the runner
# and the standalone evidence verifier use exactly the same rules.
COMPARISON_CONTRACT_FIELDS = (
    "selection_strategy",
    "eligible_layer_ids",
    "selected_layer_ids",
    "plan_id",
    "parameter_names",
    "host_bytes",
    "resident_weight_bytes",
    "backend_hbm_cache_bytes",
    "kv_reserve_bytes",
    "graph_policy",
    "workload_contract_sha256",
    "source_identity",
)


def _contract_value(payload: dict[str, Any], field: str) -> Any:
    """Read a contract field from either a flat or nested artifact document."""
    comparison = payload.get("comparison_contract")
    if isinstance(comparison, dict) and field in comparison:
        return comparison[field]
    contract = payload.get("contract")
    if isinstance(contract, dict) and field in contract:
        return contract[field]
    if field in payload:
        return payload[field]
    aliases = {
        "selected_layer_ids": ("offloaded_layer_ids",),
        "source_identity": ("source_state_sha256", "source_identity_sha256"),
    }
    for alias in aliases.get(field, ()):
        if alias in payload:
            return payload[alias]
    return None


def validate_comparable_contracts(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, Any]:
    """Validate the immutable plan and resource contract for a comparison.

    A ``ValueError`` is raised on the first mismatch so callers cannot
    accidentally publish a result with only a matching workload hash.
    """
    mismatches: list[str] = []
    for name, document in (("baseline", baseline), ("candidate", candidate)):
        cache_bytes = _contract_value(document, "backend_hbm_cache_bytes")
        if (
            not isinstance(cache_bytes, int)
            or isinstance(cache_bytes, bool)
            or cache_bytes < 0
        ):
            raise ValueError(
                f"{name} backend_hbm_cache_bytes must be a non-negative integer"
            )
    for field in COMPARISON_CONTRACT_FIELDS:
        left = _contract_value(baseline, field)
        right = _contract_value(candidate, field)
        if left != right:
            mismatches.append(field)
    baseline_reservation = _contract_value(baseline, "uva_reservation_bytes")
    candidate_reservation = _contract_value(candidate, "uva_reservation_bytes")
    if baseline_reservation is None:
        baseline_reservation = 0
    if candidate_reservation is None:
        candidate_reservation = 0
    try:
        baseline_reservation = int(baseline_reservation)
        candidate_reservation = int(candidate_reservation)
    except (TypeError, ValueError) as exc:
        raise ValueError("uva reservation bytes must be an integer") from exc
    if baseline_reservation < 0 or candidate_reservation < 0:
        raise ValueError("uva reservation bytes must be non-negative")
    if mismatches:
        raise ValueError("comparison contract differs: " + ", ".join(mismatches))
    return {
        "schema_version": 1,
        "fields": list(COMPARISON_CONTRACT_FIELDS),
        "uva_reservation_bytes": baseline_reservation,
        "candidate_reservation_bytes": candidate_reservation,
        "reservation_equal": baseline_reservation == candidate_reservation,
    }


def local_benchmark_environment(
    environment: dict[str, str] | None = None,
) -> dict[str, str]:
    result = dict(os.environ if environment is None else environment)
    for name in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        result.pop(name, None)
    result["NO_PROXY"] = "127.0.0.1,localhost"
    result["no_proxy"] = "127.0.0.1,localhost"
    return result


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def payload_sha256(payload: object) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def manifest_comparison_contract(
    manifest: OffloadManifest,
    *,
    workload_contract_sha256: str,
    source_identity: str,
    graph_policy: str,
    kv_reserve_bytes: int,
    plan: Any | None = None,
    uva_reservation_bytes: int = 0,
    backend_hbm_cache_bytes: int = 0,
) -> dict[str, Any]:
    """Render a flat comparison contract from an immutable plan or manifest.

    Schema-v1 manifests are deliberately labelled diagnostic legacy evidence;
    production runs should pass the v2 ``CudaResidencyPlan`` as ``plan``.
    """
    if plan is not None:
        selected = list(plan.offloaded_layer_ids)
        eligible = list(plan.eligible_layer_ids)
        strategy = str(plan.selection_strategy)
        plan_id = str(plan.plan_id)
        parameter_names = sorted(
            f"model.layers.{layer_id}.mlp.experts.{name}"
            for layer_id in selected
            for name in ("w13_weight", "w2_weight")
        )
        host_bytes = int(
            sum(
                item.routed_expert_bytes
                for item in plan.layer_expert_bytes
                if item.layer_id in selected
            )
        )
        resident_bytes = int(
            sum(
                item.routed_expert_bytes
                for item in plan.layer_expert_bytes
                if item.layer_id not in selected
            )
        )
    else:
        selected = list(manifest.layer_ids)
        eligible = list(manifest.layer_ids)
        strategy = "legacy_diagnostic_manifest"
        plan_id = None
        parameter_names = sorted(manifest.parameter_names)
        host_bytes = int(manifest.total_elements * 2)
        resident_bytes = 0
    return {
        "selection_strategy": strategy,
        "eligible_layer_ids": eligible,
        "selected_layer_ids": selected,
        "plan_id": plan_id,
        "parameter_names": parameter_names,
        "host_bytes": host_bytes,
        "resident_weight_bytes": resident_bytes,
        "backend_hbm_cache_bytes": int(backend_hbm_cache_bytes),
        "kv_reserve_bytes": int(kv_reserve_bytes),
        "graph_policy": graph_policy,
        "workload_contract_sha256": str(workload_contract_sha256),
        "source_identity": str(source_identity),
        "uva_reservation_bytes": int(uva_reservation_bytes),
        "legacy_evidence": plan is None,
    }


def build_server_command(
    *,
    python_executable: str,
    mode: str,
    manifest: OffloadManifest,
    host: str,
    port: int,
    served_model_name: str,
    max_num_seqs: int,
    max_model_len: int,
    max_num_batched_tokens: int,
    kv_cache_memory_bytes: int,
) -> list[str]:
    if mode not in BENCHMARK_MODES:
        raise ValueError(f"unsupported benchmark mode: {mode}")
    command = [
        str(Path(python_executable).parent / "vllm"),
        "serve",
        manifest.model.path,
        "--served-model-name",
        served_model_name,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--host",
        host,
        "--port",
        str(port),
        "--max-num-seqs",
        str(max_num_seqs),
        "--max-model-len",
        str(max_model_len),
        "--max-num-batched-tokens",
        str(max_num_batched_tokens),
        "--gpu-memory-utilization",
        "0.98",
        "--kv-cache-memory-bytes",
        str(kv_cache_memory_bytes),
        "--seed",
        "0",
        "--no-enable-prefix-caching",
        "--disable-log-stats",
        "--attention-backend",
        "FLASH_ATTN",
    ]
    if mode.startswith("uva"):
        command.extend(["--cpu-offload-gb", str(STOCK_UVA_CPU_OFFLOAD_GB)])
    if mode == "uva-full-and-piecewise":
        command.extend(
            [
                "--compilation-config",
                (
                    '{"cudagraph_mode":"FULL_AND_PIECEWISE",'
                    '"custom_ops":["+unquantized_fused_moe"]}'
                ),
                "--max-cudagraph-capture-size",
                str(max_num_seqs),
            ]
        )
    elif mode.endswith("piecewise"):
        command.extend(
            [
                "--compilation-config",
                (
                    '{"cudagraph_mode":"PIECEWISE",'
                    '"custom_ops":["+unquantized_fused_moe"]}'
                ),
                "--max-cudagraph-capture-size",
                str(max_num_seqs),
            ]
        )
    else:
        command.append("--enforce-eager")
    return command


def build_client_command(
    *,
    python_executable: str,
    base_url: str,
    served_model_name: str,
    tokenizer: str,
    dataset_path: str | Path,
    result_dir: str | Path,
    result_filename: str,
    num_prompts: int,
    output_len: int,
    max_concurrency: int,
    seed: int,
    warmup_requests: int,
    request_id_prefix: str,
    ignore_eos: bool = True,
    disable_shuffle: bool = False,
    dataset_name: str = "sharegpt",
    skip_chat_template: bool = False,
) -> list[str]:
    if dataset_name not in {"sharegpt", "custom"}:
        raise ValueError(f"unsupported benchmark dataset loader: {dataset_name}")
    command = [
        str(Path(python_executable).parent / "vllm"),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        base_url,
        "--endpoint",
        "/v1/completions",
        "--model",
        served_model_name,
        "--tokenizer",
        tokenizer,
        "--dataset-name",
        dataset_name,
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(num_prompts),
        (
            "--sharegpt-output-len"
            if dataset_name == "sharegpt"
            else "--custom-output-len"
        ),
        str(output_len),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(max_concurrency),
        "--seed",
        str(seed),
        "--num-warmups",
        str(warmup_requests),
        "--temperature",
        "0",
        "--no-oversample",
        "--disable-tqdm",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,90,99",
        "--request-id-prefix",
        request_id_prefix,
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(result_dir),
        "--result-filename",
        result_filename,
    ]
    if ignore_eos:
        command.append("--ignore-eos")
    if disable_shuffle:
        command.append("--disable-shuffle")
    if skip_chat_template:
        command.append("--skip-chat-template")
    return command


def normalize_benchmark_result(
    raw: dict[str, Any], *, expected_requests: int, expected_output_len: int | None
) -> dict[str, Any]:
    completed = raw.get("completed")
    failed = raw.get("failed", 0)
    total_output = raw.get("total_output_tokens")
    if completed != expected_requests or failed != 0:
        raise RuntimeError(
            f"incomplete benchmark: completed={completed}, failed={failed}, "
            f"expected={expected_requests}"
        )
    expected_output = (
        expected_requests * expected_output_len
        if expected_output_len is not None
        else None
    )
    if expected_output is not None and total_output != expected_output:
        raise RuntimeError(
            f"unexpected output token count: actual={total_output}, "
            f"expected={expected_output}"
        )
    if (
        not isinstance(total_output, int)
        or isinstance(total_output, bool)
        or total_output <= 0
    ):
        raise RuntimeError(f"invalid output token count: {total_output}")
    metrics: dict[str, float] = {}
    for name in SUMMARY_METRICS:
        value = raw.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise RuntimeError(f"benchmark result has no numeric {name}")
        value = float(value)
        if not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"benchmark result has invalid {name}={value}")
        metrics[name] = value
    return {
        "schema_version": 1,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": int(raw["total_input_tokens"]),
        "total_output_tokens": int(total_output),
        "duration_s": float(raw["duration"]),
        "metrics": metrics,
    }


def summarize_repetitions(
    repetitions: Sequence[dict[str, Any]],
    *,
    minimum_repetitions: int = 3,
) -> dict[str, dict[str, Any]]:
    if minimum_repetitions < 1:
        raise ValueError("minimum_repetitions must be positive")
    if len(repetitions) < minimum_repetitions:
        raise ValueError(
            f"benchmark summary requires at least {minimum_repetitions} repetitions"
        )
    summary: dict[str, dict[str, Any]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(item["metrics"][metric]) for item in repetitions]
        summary[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values) if len(values) > 1 else None,
            "min": min(values),
            "max": max(values),
        }
    return summary


def read_offload_telemetry(
    mode: str, manifest: OffloadManifest, profile_path: str | Path
) -> dict[str, Any]:
    path = Path(profile_path)
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    required_runtime_mode = None
    if mode == "uva-full-and-piecewise":
        required_runtime_mode = "FULL"
    elif mode.endswith("piecewise"):
        required_runtime_mode = "PIECEWISE"
    all_captures = [
        event for event in events if event.get("event") == "cudagraph_capture"
    ]
    all_replays = [
        event for event in events if event.get("event") == "cudagraph_replay"
    ]
    captures = [
        event
        for event in all_captures
        if event.get("runtime_mode") == required_runtime_mode
    ]
    replays = [
        event
        for event in all_replays
        if event.get("runtime_mode") == required_runtime_mode
    ]
    if required_runtime_mode and (not captures or not replays):
        raise RuntimeError(
            f"{required_runtime_mode} measurement has no CUDA graph capture and replay"
        )
    capture_modes = sorted({str(event.get("runtime_mode")) for event in all_captures})
    replay_modes = sorted({str(event.get("runtime_mode")) for event in all_replays})
    graph_telemetry = {
        "cudagraph_captures": len(captures),
        "cudagraph_replays": len(replays),
        "required_cudagraph_runtime_mode": required_runtime_mode,
        "observed_cudagraph_capture_modes": capture_modes,
        "observed_cudagraph_replay_modes": replay_modes,
    }
    if mode.startswith("uva"):
        matches = [event for event in events if event.get("event") == "stock_uva"]
        if len(matches) != 1:
            raise RuntimeError("official UVA telemetry is missing or duplicated")
        event = matches[0]
        implementation = event.get("implementation")
        expected = {
            "vllm.model_executor.offloader.uva.UVAOffloader",
            "vllm_latchmoe_cuda.uva.ManifestUVAOffloader",
        }
        if implementation not in expected:
            raise RuntimeError(f"unexpected UVA implementation: {implementation}")
        manifest_bytes = int(event["cpu_offload_bytes"])
        residual_bytes = int(event.get("residual_uva_bytes", 0))
        residual_budget = int(event.get("residual_uva_max_bytes", 0))
        return {
            "implementation": implementation,
            "manifest_bytes": manifest_bytes,
            "residual_implementation": event.get("residual_implementation"),
            "residual_uva_bytes": residual_bytes,
            "actual_offload_bytes": manifest_bytes + residual_bytes,
            "configured_budget_bytes": int(event["cpu_offload_max_bytes"])
            + residual_budget,
            "uva_reservation_bytes": int(
                event.get("uva_reservation_bytes", 0)
            ),
            **graph_telemetry,
        }
    residual = [event for event in events if event.get("event") == "residual_uva"]
    if len(residual) > 1:
        raise RuntimeError("LatchMoE residual UVA telemetry is duplicated")
    wave_events = [event for event in events if event.get("event") == "exact_waves"]
    main_cache_events = [
        event for event in events if event.get("event") == "main_cache_waves"
    ]
    if any(
        event.get("pair_layout") != "unified_token_expert_v1"
        or int(event.get("pair_layout_build_count", 0)) != 1
        or int(event.get("pair_count", 0)) <= 0
        for event in main_cache_events
    ):
        raise RuntimeError(
            "not every Main Cache wave used one unified token-expert layout"
        )
    direct_slot_events = [
        event for event in events if event.get("event") == "direct_slots"
    ]
    device_wave_events = [
        event
        for event in wave_events
        if event.get("pair_planner_mode") == "cuda_device"
        and event.get("scatter_mode") == "layer_index_add"
    ]
    full_capacity = manifest.num_slots == manifest.model.num_experts
    if required_runtime_mode is None and full_capacity:
        direct_layers = {int(event["layer_id"]) for event in direct_slot_events}
        if direct_layers != set(manifest.layer_ids):
            raise RuntimeError(
                "ShareGPT measurement did not exercise every full-capacity "
                "direct-slot layer"
            )
    elif required_runtime_mode is None and not wave_events and not main_cache_events:
        raise RuntimeError("ShareGPT measurement did not exercise exact waves or Main Cache waves")
    if wave_events and len(device_wave_events) != len(wave_events):
        raise RuntimeError("not every exact wave used the CUDA device planner")
    if any(
        int(event.get("active_experts", -1)) < 0
        or int(event.get("active_experts", -1)) > manifest.num_slots
        or int(event.get("slot_capacity", -1)) != manifest.num_slots
        for event in direct_slot_events
    ):
        raise RuntimeError("invalid full-capacity direct-slot telemetry")
    event = residual[0] if residual else None
    manifest_bytes = manifest.total_elements * 2
    residual_bytes = int(event["cpu_offload_bytes"]) if event is not None else 0
    ledger_events = [event for event in events if event.get("event") == "residency_ledger"]
    dynamic_slot_bytes = sum(
        int(event.get("dynamic_slot_bytes", 0) or 0) for event in ledger_events
    )
    return {
        "implementation": "vllm_latchmoe_cuda.offloader.CudaSEWOffloader",
        "residual_implementation": "vllm.model_executor.offloader.uva.UVAOffloader",
        "manifest_bytes": manifest_bytes,
        "dynamic_slot_bytes": dynamic_slot_bytes,
        "residual_uva_bytes": residual_bytes,
        "actual_offload_bytes": manifest_bytes + residual_bytes,
        "configured_budget_bytes": manifest_bytes
        + (int(event["cpu_offload_max_bytes"]) if event is not None else 0),
        "exact_wave_events": len(wave_events),
        "main_cache_wave_events": len(main_cache_events),
        "unified_pair_layout_events": len(main_cache_events),
        "cuda_device_planner_events": len(device_wave_events),
        "direct_slot_events": len(direct_slot_events),
        **graph_telemetry,
    }


def compare_mode_summaries(
    uva: dict[str, Any],
    latchmoe: dict[str, Any],
    *,
    require_equal_offload_bytes: bool = True,
    require_equal_contract: bool = True,
) -> dict[str, Any]:
    reference_mode = str(uva.get("mode", ""))
    candidate_mode = str(latchmoe.get("mode", ""))
    if not reference_mode.startswith("uva"):
        raise ValueError("reference summary must be official UVA")
    if not candidate_mode.startswith("latchmoe-"):
        raise ValueError("candidate summary must be LatchMoE")
    if reference_mode.endswith("piecewise") != candidate_mode.endswith("piecewise"):
        raise ValueError("benchmark graph policy differs")
    if uva.get("workload_contract_sha256") != latchmoe.get("workload_contract_sha256"):
        raise ValueError("benchmark workload contracts differ")
    uva_bytes = uva["offload_telemetry"]["actual_offload_bytes"]
    latchmoe_bytes = latchmoe["offload_telemetry"]["actual_offload_bytes"]
    if require_equal_offload_bytes and uva_bytes != latchmoe_bytes:
        raise ValueError(
            f"actual offload bytes differ: UVA={uva_bytes}, LatchMoE={latchmoe_bytes}"
        )

    # New artifacts carry the full plan/resource contract.  Keep the legacy
    # summary shape readable for historical reports, but enforce every field
    # whenever either side advertises the contract.
    if any(
        _contract_value(uva, field) is not None
        or _contract_value(latchmoe, field) is not None
        for field in COMPARISON_CONTRACT_FIELDS
    ):
        if _contract_value(uva, "legacy_evidence") is True or _contract_value(
            latchmoe, "legacy_evidence"
        ) is True:
            raise ValueError("legacy temporary-bank/shared-pool evidence is not comparable")
        if require_equal_contract:
            validate_comparable_contracts(uva, latchmoe)

    comparison: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        reference = float(uva["metrics"][metric]["median"])
        candidate = float(latchmoe["metrics"][metric]["median"])
        if metric.endswith("throughput"):
            change = (candidate / reference - 1.0) * 100.0
            direction = "higher_is_better"
        else:
            change = (reference - candidate) / reference * 100.0
            direction = "lower_is_better"
        comparison[metric] = {
            "uva": reference,
            "latchmoe": candidate,
            "improvement_percent": change,
            "direction": direction,
        }
    return {
        "schema_version": 1,
        "reference_mode": reference_mode,
        "candidate_mode": candidate_mode,
        "workload_contract_sha256": uva["workload_contract_sha256"],
        "actual_offload_bytes": uva_bytes,
        "latchmoe_actual_offload_bytes": latchmoe_bytes,
        "comparable_offload_bytes": uva_bytes == latchmoe_bytes,
        "comparable_contract": (
            True
            if not any(
                _contract_value(uva, field) is not None
                or _contract_value(latchmoe, field) is not None
                for field in COMPARISON_CONTRACT_FIELDS
            )
            else all(
                _contract_value(uva, field) == _contract_value(latchmoe, field)
                for field in COMPARISON_CONTRACT_FIELDS
            )
        ),
        "comparison_contract": {
            field: _contract_value(uva, field)
            for field in COMPARISON_CONTRACT_FIELDS
            if _contract_value(uva, field) is not None
        },
        "metrics": comparison,
    }


def _compare_execution_policies(
    eager: dict[str, Any], piecewise: dict[str, Any]
) -> dict[str, Any]:
    comparison: dict[str, Any] = {}
    for metric in SUMMARY_METRICS:
        eager_value = float(eager["metrics"][metric]["median"])
        piecewise_value = float(piecewise["metrics"][metric]["median"])
        if metric.endswith("throughput"):
            change = (piecewise_value / eager_value - 1.0) * 100.0
            direction = "higher_is_better"
        else:
            change = (eager_value - piecewise_value) / eager_value * 100.0
            direction = "lower_is_better"
        comparison[metric] = {
            "eager": eager_value,
            "piecewise": piecewise_value,
            "improvement_percent": change,
            "direction": direction,
        }
    return {
        "eager_mode": eager["mode"],
        "piecewise_mode": piecewise["mode"],
        "metrics": comparison,
    }


def compare_ablation_summaries(
    *,
    uva_eager: dict[str, Any],
    uva_piecewise: dict[str, Any],
    latchmoe_eager: dict[str, Any],
    latchmoe_piecewise: dict[str, Any],
) -> dict[str, Any]:
    summaries = {
        "uva": uva_eager,
        "uva-piecewise": uva_piecewise,
        "latchmoe-eager": latchmoe_eager,
        "latchmoe-piecewise": latchmoe_piecewise,
    }
    for expected_mode, summary in summaries.items():
        if summary.get("mode") != expected_mode:
            raise ValueError(
                f"expected {expected_mode} summary, got {summary.get('mode')!r}"
            )

    workload_hashes = {
        summary.get("workload_contract_sha256") for summary in summaries.values()
    }
    if len(workload_hashes) != 1 or None in workload_hashes:
        raise ValueError("2x2 benchmark workload contracts differ")
    source_hashes = {
        summary.get("source_state_sha256") for summary in summaries.values()
    }
    if len(source_hashes) != 1 or None in source_hashes:
        raise ValueError("2x2 benchmark source states differ")
    git_commits = {summary.get("git_commit") for summary in summaries.values()}
    if len(git_commits) != 1 or None in git_commits:
        raise ValueError("2x2 benchmark git commits differ")
    offload_bytes = {
        int(summary["offload_telemetry"]["actual_offload_bytes"])
        for summary in summaries.values()
    }
    if len(offload_bytes) != 1:
        raise ValueError("2x2 benchmark actual offload bytes differ")

    return {
        "schema_version": 1,
        "design": "offloader_x_execution_policy_2x2",
        "workload_contract_sha256": workload_hashes.pop(),
        "source_state_sha256": source_hashes.pop(),
        "git_commit": git_commits.pop(),
        "actual_offload_bytes": offload_bytes.pop(),
        "latchmoe_effect": {
            "eager": compare_mode_summaries(uva_eager, latchmoe_eager),
            "piecewise": compare_mode_summaries(uva_piecewise, latchmoe_piecewise),
        },
        "graph_effect": {
            "uva": _compare_execution_policies(uva_eager, uva_piecewise),
            "latchmoe": _compare_execution_policies(latchmoe_eager, latchmoe_piecewise),
        },
    }
