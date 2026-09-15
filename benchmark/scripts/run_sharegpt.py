#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from vllm_latchmoe_cuda.artifacts import ArtifactRun, RunKind
from vllm_latchmoe_cuda.benchmark import (
    BENCHMARK_MODES,
    build_client_command,
    build_server_command,
    file_sha256,
    local_benchmark_environment,
    manifest_comparison_contract,
    normalize_benchmark_result,
    payload_sha256,
    read_offload_telemetry,
    summarize_repetitions,
)
from vllm_latchmoe_cuda.manifest import OffloadManifest
from vllm_latchmoe_cuda.manifest import build_identity_lock, serialize_identity_lock
from vllm_latchmoe_cuda.residency_plan import deserialize_residency_plan


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = (
    ROOT / "benchmark/manifests/"
    "offload_manifest.qwen3-base-ad44.first12.graph128.local.json"
)
DEFAULT_DATASET = Path("/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")
SHAREGPT_REVISION = "192ab2185289094fc556ec8ce5ce1e8e587154ca"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated 50-request ShareGPT measurements against one backend"
    )
    parser.add_argument("--mode", required=True, choices=BENCHMARK_MODES)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--plan",
        type=Path,
        help="immutable residency plan JSON; required for qualified production evidence",
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--exploratory",
        action="store_true",
        help="allow exactly one repetition without promoting it to a final result",
    )
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-requests", type=int, default=2)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8026)
    parser.add_argument("--startup-timeout-s", type=float, default=600.0)
    parser.add_argument("--shutdown-timeout-s", type=float, default=30.0)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    return parser.parse_args()


def _wait_for_server(url: str, process: subprocess.Popen, timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    request = urllib.request.Request(url, headers={"Connection": "close"})
    last_error = ""
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            with opener.open(request, timeout=5) as response:
                if 200 <= int(response.status) < 500:
                    return
        except (urllib.error.URLError, TimeoutError, OSError) as error:
            last_error = str(error)
        time.sleep(2)
    raise TimeoutError(f"server did not become ready: {last_error}")


def _ensure_port_is_free(host: str, port: int) -> None:
    with socket.socket() as probe:
        probe.settimeout(0.2)
        if probe.connect_ex((host, port)) == 0:
            raise RuntimeError(f"benchmark port is already in use: {host}:{port}")


def _terminate(process: subprocess.Popen, timeout_s: float) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


def _server_environment(
    mode: str,
    manifest_path: Path,
    profile_path: Path,
    *,
    plan_json: str | None = None,
    identity_lock_json: str | None = None,
) -> dict[str, str]:
    environment = os.environ.copy()
    for name in (
        "VLLM_LATCHMOE_MODE",
        "VLLM_LATCHMOE_MANIFEST",
        "VLLM_LATCHMOE_PROFILE_PATH",
        "VLLM_LATCHMOE_TELEMETRY_PATH",
        "VLLM_LATCHMOE_GRAPH_MODE",
        "VLLM_LATCHMOE_OVERLAP",
        "VLLM_LATCHMOE_WAVE_SLOTS",
        "VLLM_LATCHMOE_RESIDENCY_PLAN_JSON",
        "VLLM_LATCHMOE_IDENTITY_LOCK_JSON",
    ):
        environment.pop(name, None)
    environment.update(
        {
            "VLLM_PLUGINS": "latchmoe_cuda",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if mode.startswith("uva"):
        environment.update(
            {
                "VLLM_LATCHMOE_MODE": "uva",
                "VLLM_LATCHMOE_MANIFEST": str(manifest_path),
                "VLLM_LATCHMOE_TELEMETRY_PATH": str(profile_path),
            }
        )
        if plan_json is not None:
            environment["VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"] = plan_json
        if identity_lock_json is not None:
            environment["VLLM_LATCHMOE_IDENTITY_LOCK_JSON"] = identity_lock_json
    else:
        environment.update(
            {
                "VLLM_LATCHMOE_MODE": "latchmoe",
                "VLLM_LATCHMOE_MANIFEST": str(manifest_path),
                "VLLM_LATCHMOE_PROFILE_PATH": str(profile_path),
                "VLLM_LATCHMOE_OVERLAP": "1",
            }
        )
        if plan_json is not None:
            environment["VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"] = plan_json
        if identity_lock_json is not None:
            environment["VLLM_LATCHMOE_IDENTITY_LOCK_JSON"] = identity_lock_json
        if mode.endswith("piecewise"):
            environment["VLLM_LATCHMOE_GRAPH_MODE"] = "piecewise"
    return environment


def _write_child_manifest(
    run: ArtifactRun,
    *,
    relative_dir: Path,
    run_id: str,
    contract_sha256: str,
    source_state_sha256: str,
    result_path: Path,
) -> str:
    result_sha256 = hashlib.sha256(result_path.read_bytes()).hexdigest()
    relative_manifest = relative_dir / "run_manifest.json"
    run.write_json(
        relative_manifest.as_posix(),
        {
            "schema_version": 1,
            "kind": "measurement",
            "status": "completed",
            "exit_code": 0,
            "final_result": False,
            "run_id": run_id,
            "contract_sha256": contract_sha256,
            "source_state_sha256": source_state_sha256,
            "result_artifact": result_path.name,
            "result_sha256": result_sha256,
        },
    )
    return relative_manifest.as_posix()


def execute(args: argparse.Namespace, run: ArtifactRun) -> None:
    if args.exploratory and args.repetitions != 1:
        raise ValueError("exploratory measurements require exactly 1 repetition")
    if not args.exploratory and args.repetitions < 3:
        raise ValueError("final measurements require at least 3 repetitions")
    if args.num_prompts != 50:
        raise ValueError("the frozen ShareGPT contract requires exactly 50 prompts")
    if not args.exploratory and args.plan is None:
        raise ValueError("final measurements require an immutable --plan")
    for name in ("output_len", "max_concurrency", "max_num_seqs"):
        if int(getattr(args, name)) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not args.dataset.is_file():
        raise FileNotFoundError(args.dataset)
    _ensure_port_is_free(args.host, args.port)

    manifest_path = args.manifest.resolve()
    dataset_path = args.dataset.resolve()
    manifest = OffloadManifest.load(manifest_path)
    manifest.validate_model_files()
    plan = None
    plan_json = None
    identity_lock_json = None
    if args.plan is not None:
        plan = deserialize_residency_plan(args.plan.read_text(encoding="utf-8"))
        if tuple(manifest.layer_ids) != tuple(plan.offloaded_layer_ids):
            raise ValueError("manifest layers must exactly match plan offloaded layers")
        plan_json = json.dumps(plan.to_jsonable(), sort_keys=True, separators=(",", ":"))
        identity_lock_json = serialize_identity_lock(
            build_identity_lock(
                plan,
                ["--model", manifest.model.path, "--revision", manifest.model.revision],
            )
        )
    dataset_sha256 = file_sha256(dataset_path)
    run_manifest = json.loads((run.path / "run_manifest.json").read_text())
    source_state_sha256 = run_manifest["source_state_sha256"]
    git_commit = run_manifest["git_commit"]
    workload_contract = {
        "schema_version": 1,
        "model_path": manifest.model.path,
        "model_revision": manifest.model.revision,
        "manifest_sha256": manifest.manifest_sha256,
        "vllm_version": manifest.vllm.version,
        "dtype": manifest.dtype,
        "tensor_parallel_size": manifest.tensor_parallel_size,
        "dataset": "ShareGPT_V3_unfiltered_cleaned_split.json",
        "dataset_revision": SHAREGPT_REVISION,
        "dataset_sha256": dataset_sha256,
        "dataset_bytes": dataset_path.stat().st_size,
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "ignore_eos": True,
        "seed": args.seed,
        "request_rate": "inf",
        "max_concurrency": args.max_concurrency,
        "warmup_requests": args.warmup_requests,
        "repetitions": args.repetitions,
        "exploratory": args.exploratory,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "gpu_memory_utilization": 0.98,
        "prefix_caching": False,
        "server_log_stats": False,
        "compile_cache": False,
    }
    workload_contract_sha256 = payload_sha256(workload_contract)
    mode_contract = {
        **workload_contract,
        "mode": args.mode,
        "overlap_enabled": args.mode.startswith("latchmoe"),
        "latchmoe_wave_slots": (
            min(manifest.num_slots, 32)
            if args.mode.startswith("latchmoe")
            and manifest.num_slots < manifest.model.num_experts
            else None
        ),
    }
    mode_contract.update(
        manifest_comparison_contract(
            manifest,
            workload_contract_sha256=workload_contract_sha256,
            source_identity=source_state_sha256,
            graph_policy=("piecewise" if args.mode.endswith("piecewise") else "eager"),
            kv_reserve_bytes=args.kv_cache_memory_bytes,
            plan=plan,
        )
    )
    mode_contract_sha256 = payload_sha256(mode_contract)
    run.write_json(
        "contract.json",
        {
            **mode_contract,
            "workload_contract_sha256": workload_contract_sha256,
            "mode_contract_sha256": mode_contract_sha256,
        },
    )

    served_model_name = "latchmoe-qwen3-30b-a3b"
    profile_path = (run.path / "profile.jsonl").resolve()
    server_command = build_server_command(
        python_executable=sys.executable,
        mode=args.mode,
        manifest=manifest,
        host=args.host,
        port=args.port,
        served_model_name=served_model_name,
        max_num_seqs=args.max_num_seqs,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
    )
    run.write_json("server_command.json", server_command)
    environment = _server_environment(
        args.mode,
        manifest_path,
        profile_path,
        plan_json=plan_json,
        identity_lock_json=identity_lock_json,
    )

    server_log = (run.path / "server.log").open("w", encoding="utf-8")
    process = subprocess.Popen(
        server_command,
        cwd=ROOT,
        env=environment,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    repetitions: list[dict[str, object]] = []
    child_manifests: list[str] = []
    try:
        base_url = f"http://{args.host}:{args.port}"
        _wait_for_server(
            f"{base_url}/v1/models", process, float(args.startup_timeout_s)
        )
        for index in range(args.repetitions):
            relative_dir = Path(f"repetition-{index + 1:02d}")
            repetition_dir = run.path / relative_dir
            repetition_dir.mkdir()
            raw_result_path = repetition_dir / "raw_result.json"
            client_command = build_client_command(
                python_executable=sys.executable,
                base_url=base_url,
                served_model_name=served_model_name,
                tokenizer=manifest.model.path,
                dataset_path=dataset_path,
                result_dir=repetition_dir,
                result_filename=raw_result_path.name,
                num_prompts=args.num_prompts,
                output_len=args.output_len,
                max_concurrency=args.max_concurrency,
                seed=args.seed,
                warmup_requests=args.warmup_requests,
                request_id_prefix=f"{args.mode}-rep-{index + 1}-",
            )
            run.write_json(
                (relative_dir / "client_command.json").as_posix(), client_command
            )
            with (repetition_dir / "client.log").open(
                "w", encoding="utf-8"
            ) as client_log:
                completed = subprocess.run(
                    client_command,
                    cwd=ROOT,
                    env=local_benchmark_environment(),
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                    text=True,
                    check=False,
                )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"benchmark repetition {index + 1} exited "
                    f"with code {completed.returncode}"
                )
            if not raw_result_path.is_file():
                raise RuntimeError(
                    f"benchmark repetition {index + 1} produced no raw result"
                )
            raw = json.loads(raw_result_path.read_text(encoding="utf-8"))
            normalized = normalize_benchmark_result(
                raw,
                expected_requests=args.num_prompts,
                expected_output_len=args.output_len,
            )
            normalized["run_id"] = f"{args.mode}-rep-{index + 1}"
            measurement_path = run.write_json(
                (relative_dir / "measurement.json").as_posix(), normalized
            )
            repetitions.append(normalized)
            child_manifests.append(
                _write_child_manifest(
                    run,
                    relative_dir=relative_dir,
                    run_id=f"{args.mode}-rep-{index + 1}",
                    contract_sha256=mode_contract_sha256,
                    source_state_sha256=source_state_sha256,
                    result_path=measurement_path,
                )
            )
    finally:
        _terminate(process, float(args.shutdown_timeout_s))
        server_log.close()

    telemetry = read_offload_telemetry(args.mode, manifest, profile_path)
    summary = {
        "schema_version": 1,
        "mode": args.mode,
        "workload_contract_sha256": workload_contract_sha256,
        "mode_contract_sha256": mode_contract_sha256,
        "source_state_sha256": source_state_sha256,
        "git_commit": git_commit,
        "dataset_sha256": dataset_sha256,
        "repetitions": args.repetitions,
        "exploratory": args.exploratory,
        "final_result": not args.exploratory,
        "overlap_enabled": args.mode.startswith("latchmoe"),
        "offload_telemetry": telemetry,
        "comparison_contract": {
            key: mode_contract[key]
            for key in (
                "selection_strategy",
                "eligible_layer_ids",
                "selected_layer_ids",
                "plan_id",
                "parameter_names",
                "host_bytes",
                "resident_weight_bytes",
                "kv_reserve_bytes",
                "graph_policy",
                "workload_contract_sha256",
                "source_identity",
                "uva_reservation_bytes",
                "legacy_evidence",
            )
        },
        "metrics": summarize_repetitions(
            repetitions, minimum_repetitions=1 if args.exploratory else 3
        ),
    }
    run.write_json("summary.json", summary)
    run.record_completion(exit_code=0)
    if not args.exploratory:
        run.mark_final(repetition_manifests=child_manifests)


def main() -> int:
    args = parse_args()
    run = ArtifactRun.create(
        args.artifact_dir,
        kind=RunKind.MEASUREMENT,
        command=[sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]],
    )
    try:
        execute(args, run)
        exit_code = 0
    except BaseException as error:
        exit_code = getattr(error, "returncode", 1) or 1
        run.record_failure(error, exit_code=exit_code)
    finally:
        run.write_inventory()
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
