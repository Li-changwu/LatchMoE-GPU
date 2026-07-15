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


BENCHMARK_MODES = ("uva", "latchmoe-eager", "latchmoe-piecewise")
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
    ]
    if mode == "uva":
        command.extend(["--cpu-offload-gb", str(STOCK_UVA_CPU_OFFLOAD_GB)])
    if mode != "latchmoe-piecewise":
        command.append("--enforce-eager")
    else:
        command.extend(["--compilation-config", '{"cudagraph_mode":"PIECEWISE"}'])
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
) -> list[str]:
    return [
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
        "sharegpt",
        "--dataset-path",
        str(dataset_path),
        "--num-prompts",
        str(num_prompts),
        "--sharegpt-output-len",
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
        "--ignore-eos",
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


def normalize_benchmark_result(
    raw: dict[str, Any], *, expected_requests: int, expected_output_len: int
) -> dict[str, Any]:
    completed = raw.get("completed")
    failed = raw.get("failed", 0)
    total_output = raw.get("total_output_tokens")
    if completed != expected_requests or failed != 0:
        raise RuntimeError(
            f"incomplete benchmark: completed={completed}, failed={failed}, "
            f"expected={expected_requests}"
        )
    expected_output = expected_requests * expected_output_len
    if total_output != expected_output:
        raise RuntimeError(
            f"unexpected output token count: actual={total_output}, "
            f"expected={expected_output}"
        )
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
) -> dict[str, dict[str, Any]]:
    if len(repetitions) < 3:
        raise ValueError("final benchmark summaries require at least 3 repetitions")
    summary: dict[str, dict[str, Any]] = {}
    for metric in SUMMARY_METRICS:
        values = [float(item["metrics"][metric]) for item in repetitions]
        summary[metric] = {
            "values": values,
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "stdev": statistics.stdev(values),
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
    if mode == "uva":
        matches = [event for event in events if event.get("event") == "stock_uva"]
        if len(matches) != 1:
            raise RuntimeError("official UVA telemetry is missing or duplicated")
        event = matches[0]
        implementation = event.get("implementation")
        expected = "vllm.model_executor.offloader.uva.UVAOffloader"
        if implementation != expected:
            raise RuntimeError(f"unexpected UVA implementation: {implementation}")
        return {
            "implementation": implementation,
            "actual_offload_bytes": int(event["cpu_offload_bytes"]),
            "configured_budget_bytes": int(event["cpu_offload_max_bytes"]),
        }
    residual = [event for event in events if event.get("event") == "residual_uva"]
    if len(residual) != 1:
        raise RuntimeError("LatchMoE residual UVA telemetry is missing or duplicated")
    wave_events = [event for event in events if event.get("event") == "exact_waves"]
    device_wave_events = [
        event
        for event in wave_events
        if event.get("pair_planner_mode") == "cuda_device"
        and event.get("scatter_mode") == "layer_index_add"
    ]
    if not wave_events:
        raise RuntimeError("ShareGPT measurement did not exercise exact waves")
    if wave_events and len(device_wave_events) != len(wave_events):
        raise RuntimeError("not every exact wave used the CUDA device planner")
    event = residual[0]
    manifest_bytes = manifest.total_elements * 2
    residual_bytes = int(event["cpu_offload_bytes"])
    return {
        "implementation": "vllm_latchmoe_cuda.offloader.CudaSEWOffloader",
        "residual_implementation": "vllm.model_executor.offloader.uva.UVAOffloader",
        "manifest_bytes": manifest_bytes,
        "residual_uva_bytes": residual_bytes,
        "actual_offload_bytes": manifest_bytes + residual_bytes,
        "configured_budget_bytes": manifest_bytes + int(event["cpu_offload_max_bytes"]),
        "exact_wave_events": len(wave_events),
        "cuda_device_planner_events": len(device_wave_events),
    }


def compare_mode_summaries(
    uva: dict[str, Any], latchmoe: dict[str, Any]
) -> dict[str, Any]:
    if uva.get("mode") != "uva":
        raise ValueError("reference summary must be official UVA")
    if not str(latchmoe.get("mode", "")).startswith("latchmoe-"):
        raise ValueError("candidate summary must be LatchMoE")
    if uva.get("workload_contract_sha256") != latchmoe.get("workload_contract_sha256"):
        raise ValueError("benchmark workload contracts differ")
    uva_bytes = uva["offload_telemetry"]["actual_offload_bytes"]
    latchmoe_bytes = latchmoe["offload_telemetry"]["actual_offload_bytes"]
    if uva_bytes != latchmoe_bytes:
        raise ValueError(
            f"actual offload bytes differ: UVA={uva_bytes}, LatchMoE={latchmoe_bytes}"
        )

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
        "reference_mode": "uva",
        "candidate_mode": latchmoe["mode"],
        "workload_contract_sha256": uva["workload_contract_sha256"],
        "actual_offload_bytes": uva_bytes,
        "metrics": comparison,
    }
