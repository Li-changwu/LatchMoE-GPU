#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from vllm_latchmoe_cuda.artifacts import ArtifactRun, RunKind
from vllm_latchmoe_cuda.correctness import (
    CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE,
    CORRECTNESS_MAX_NUM_SEQS,
    STOCK_UVA_CPU_OFFLOAD_GB,
    CorrectnessMismatchError,
    CorrectnessMode,
    compare_greedy_results,
    load_prompts,
    require_greedy_match,
    resolve_correctness_mode,
    run_vllm_greedy,
)
from vllm_latchmoe_cuda.manifest import OffloadManifest
from vllm_latchmoe_cuda.manifest import build_identity_lock, serialize_identity_lock
from vllm_latchmoe_cuda.residency_plan import (
    build_residency_plan,
    deserialize_residency_plan,
    serialize_residency_plan,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "benchmark/manifests/offload_manifest.json"


class WorkerProcessError(RuntimeError):
    def __init__(self, exit_code: int, stderr_path: Path):
        super().__init__(
            f"correctness worker exited with code {exit_code}; see {stderr_path}"
        )
        self.exit_code = exit_code


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a labeled Qwen greedy correctness artifact"
    )
    parser.add_argument(
        "--mode",
        required=True,
        choices=(
            "native",
            "uva",
            "uva-exact",
            "latchmoe-eager",
            "latchmoe-async",
            "latchmoe-piecewise",
            "latchmoe-waves",
        ),
    )
    parser.add_argument(
        "--kind", choices=tuple(kind.value for kind in RunKind), required=True
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument(
        "--offload-gib",
        type=float,
        default=13.5,
        help="requested routed-expert residency budget used to rebuild the plan",
    )
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--max-num-seqs", type=int, default=CORRECTNESS_MAX_NUM_SEQS)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024 * 1024)
    parser.add_argument(
        "--diagnostic-residual-uva-bytes",
        type=int,
        default=0,
        help="additional stock-UVA budget required by constrained GPUs",
    )
    return parser


def _worker_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--result-json", type=Path, required=True)
    parser.add_argument("--profile-jsonl", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, required=True)
    parser.add_argument("--offload-gib", type=float, required=True)
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--max-num-seqs", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
    parser.add_argument("--diagnostic-residual-uva-bytes", type=int, required=True)
    return parser


def _worker_main(argv: list[str]) -> int:
    args = _worker_parser().parse_args(argv)
    manifest = OffloadManifest.load(args.manifest)
    mode = resolve_correctness_mode(args.mode)
    result = run_vllm_greedy(
        manifest=manifest,
        mode=mode,
        prompts=load_prompts(args.prompts_json),
        max_tokens=args.max_tokens,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_memory_bytes=args.kv_cache_memory_bytes,
    )
    args.result_json.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if mode.expect_waves:
        events = []
        if args.profile_jsonl.exists():
            events = [
                json.loads(line)
                for line in args.profile_jsonl.read_text(encoding="utf-8").splitlines()
                if line
            ]
        if not any(event.get("event") == "exact_waves" for event in events):
            raise RuntimeError(
                "latchmoe-waves run produced no exact_waves profile event"
            )
    events = _profile_events(args.profile_jsonl)
    overlap_events = [
        event for event in events if event.get("event") == "main_cache_overlap"
    ]
    if mode.overlap_enabled and not any(
        event.get("actual_overlap") is True for event in overlap_events
    ):
        raise RuntimeError("LatchMoE async run produced no actual overlap evidence")
    if mode.backend == "latchmoe" and not mode.overlap_enabled and overlap_events:
        raise RuntimeError("serial LatchMoE run unexpectedly produced overlap evidence")
    return 0


def _environment() -> dict[str, object]:
    snapshot: dict[str, object] = {
        "python": sys.version,
        "executable": sys.executable,
        "environment": {
            key: os.environ.get(key)
            for key in (
                "CUDA_VISIBLE_DEVICES",
                "VLLM_PLUGINS",
                "VLLM_BATCH_INVARIANT",
                "HF_HUB_OFFLINE",
            )
        },
    }
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,memory.free,driver_version",
            "--format=csv,noheader,nounits",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    snapshot["nvidia_smi_exit_code"] = query.returncode
    snapshot["nvidia_smi"] = query.stdout.splitlines()
    return snapshot


def _profile_events(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]


def _offload_telemetry(
    mode: CorrectnessMode, manifest: OffloadManifest, profile_path: Path
) -> dict[str, object]:
    events = _profile_events(profile_path)
    if mode.name == "native":
        return {
            "schema_version": 1,
            "mode": mode.name,
            "implementation": "vllm-native-full-resident",
            "manifest_bytes": 0,
            "residual_uva_bytes": 0,
            "actual_offload_bytes": 0,
            "configured_budget_bytes": 0,
        }
    if mode.name in {"uva", "uva-exact"}:
        matches = [event for event in events if event.get("event") == "stock_uva"]
        if len(matches) != 1:
            raise RuntimeError("stock UVA run produced no unique stock_uva event")
        event = matches[0]
        implementation = event.get("implementation")
        if (
            mode.name == "uva"
            and implementation
            != "vllm.model_executor.offloader.uva.UVAOffloader"
        ):
            raise RuntimeError(f"unexpected UVA implementation: {implementation}")
        if mode.name == "uva-exact" and (
            implementation != "vllm_latchmoe_cuda.uva.ManifestUVAOffloader"
            or event.get("selection") != "manifest_exact"
        ):
            raise RuntimeError(
                f"unexpected exact UVA implementation: {implementation}"
            )
        manifest_bytes = int(event["cpu_offload_bytes"])
        residual_bytes = int(event.get("residual_uva_bytes", 0))
        residual_budget = int(event.get("residual_uva_max_bytes", 0))
        return {
            "schema_version": 1,
            "mode": mode.name,
            "implementation": implementation,
            "manifest_bytes": manifest_bytes,
            "residual_implementation": event.get("residual_implementation"),
            "residual_uva_bytes": residual_bytes,
            "actual_offload_bytes": manifest_bytes + residual_bytes,
            "configured_budget_bytes": int(event["cpu_offload_max_bytes"])
            + residual_budget,
            "uva_reservation_bytes": int(event.get("uva_reservation_bytes", 0)),
        }
    matches = [event for event in events if event.get("event") == "residual_uva"]
    if len(matches) > 1:
        raise RuntimeError("LatchMoE run produced multiple residual_uva events")
    # Production residency plans are CPU-first for selected routed experts and
    # explicitly forbid a residual UVA offloader.  Legacy diagnostic manifests
    # still emit the residual_uva event and retain that accounting below.
    event = matches[0] if matches else None
    manifest_bytes = manifest.total_elements * 2
    residual_bytes = int(event["cpu_offload_bytes"]) if event is not None else 0
    graph_telemetry: dict[str, object] = {}
    if mode.name == "latchmoe-piecewise":
        captures = [
            item
            for item in events
            if item.get("event") == "cudagraph_capture"
            and item.get("runtime_mode") == "PIECEWISE"
        ]
        replays = [
            item
            for item in events
            if item.get("event") == "cudagraph_replay"
            and item.get("runtime_mode") == "PIECEWISE"
        ]
        if not captures or not replays:
            raise RuntimeError(
                "latchmoe-piecewise correctness produced no PIECEWISE "
                "CUDA graph capture and replay evidence"
            )
        graph_telemetry = {
            "cudagraph_captures": len(captures),
            "cudagraph_replays": len(replays),
            "cudagraph_runtime_mode": "PIECEWISE",
        }
    return {
        "schema_version": 1,
        "mode": mode.name,
        "implementation": "vllm_latchmoe_cuda.offloader.CudaSEWOffloader",
        "residual_implementation": "vllm.model_executor.offloader.uva.UVAOffloader",
        "manifest_bytes": manifest_bytes,
        "residual_uva_bytes": residual_bytes,
        "actual_offload_bytes": manifest_bytes + residual_bytes,
        "configured_budget_bytes": (
            manifest_bytes
            + (int(event["cpu_offload_max_bytes"]) if event is not None else 0)
        ),
        **graph_telemetry,
    }


def _worker_environment(
    mode: CorrectnessMode,
    manifest_path: Path,
    profile_path: Path,
    *,
    plan_json: str | None = None,
    identity_lock_json: str | None = None,
    diagnostic_residual_uva_bytes: int = 0,
    uva_reservation_bytes: int = 0,
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "VLLM_LATCHMOE_MODE",
        "VLLM_LATCHMOE_MANIFEST",
        "VLLM_LATCHMOE_PROFILE_PATH",
        "VLLM_LATCHMOE_TELEMETRY_PATH",
        "VLLM_LATCHMOE_GRAPH_MODE",
        "VLLM_LATCHMOE_WAVE_SLOTS",
        "VLLM_LATCHMOE_RESIDENCY_PLAN_JSON",
        "VLLM_LATCHMOE_IDENTITY_LOCK_JSON",
        "VLLM_LATCHMOE_OVERLAP",
        "VLLM_LATCHMOE_DIAGNOSTIC_RESIDUAL_UVA_BYTES",
        "VLLM_LATCHMOE_UVA_RESERVATION_BYTES",
        "VLLM_BATCH_INVARIANT",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "VLLM_PLUGINS": "latchmoe_cuda",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
            "VLLM_BATCH_INVARIANT": "1",
            "PYTORCH_ALLOC_CONF": "expandable_segments:True",
        }
    )
    if mode.backend == "latchmoe":
        environment.update(
            {
                "VLLM_LATCHMOE_MODE": mode.backend,
                "VLLM_LATCHMOE_MANIFEST": str(manifest_path),
                "VLLM_LATCHMOE_PROFILE_PATH": str(profile_path),
                "VLLM_LATCHMOE_OVERLAP": "1" if mode.overlap_enabled else "0",
            }
        )
        if plan_json is not None:
            environment["VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"] = plan_json
        if identity_lock_json is not None:
            environment["VLLM_LATCHMOE_IDENTITY_LOCK_JSON"] = identity_lock_json
        if mode.name == "latchmoe-piecewise":
            environment["VLLM_LATCHMOE_GRAPH_MODE"] = "piecewise"
        else:
            environment["VLLM_LATCHMOE_GRAPH_MODE"] = "eager"
        if diagnostic_residual_uva_bytes:
            environment["VLLM_LATCHMOE_DIAGNOSTIC_RESIDUAL_UVA_BYTES"] = str(
                diagnostic_residual_uva_bytes
            )
    elif mode.name == "uva-exact":
        if plan_json is None or identity_lock_json is None:
            raise RuntimeError(
                "exact UVA worker requires parent plan and identity lock"
            )
        environment.update(
            {
                "VLLM_LATCHMOE_MODE": "uva",
                "VLLM_LATCHMOE_MANIFEST": str(manifest_path),
                "VLLM_LATCHMOE_TELEMETRY_PATH": str(profile_path),
                "VLLM_LATCHMOE_RESIDENCY_PLAN_JSON": plan_json,
                "VLLM_LATCHMOE_IDENTITY_LOCK_JSON": identity_lock_json,
            }
        )
        if diagnostic_residual_uva_bytes:
            environment["VLLM_LATCHMOE_DIAGNOSTIC_RESIDUAL_UVA_BYTES"] = str(
                diagnostic_residual_uva_bytes
            )
        if uva_reservation_bytes:
            environment["VLLM_LATCHMOE_UVA_RESERVATION_BYTES"] = str(
                uva_reservation_bytes
            )
    else:
        environment["VLLM_LATCHMOE_TELEMETRY_PATH"] = str(profile_path)
    return environment


def _execute_driver(args, run: ArtifactRun) -> None:
    if args.max_num_seqs <= 0:
        raise ValueError("max num seqs must be positive")
    if args.diagnostic_residual_uva_bytes < 0:
        raise ValueError("diagnostic residual UVA bytes must be non-negative")
    if args.diagnostic_residual_uva_bytes and args.mode not in {
        "uva-exact",
        "latchmoe-eager",
        "latchmoe-async",
    }:
        raise ValueError(
            "diagnostic residual UVA is only supported by exact-UVA, serial, "
            "and async correctness modes"
        )
    manifest = OffloadManifest.load(args.manifest)
    manifest.validate_model_files()
    mode = resolve_correctness_mode(args.mode)
    plan_json = None
    identity_lock_json = None
    backend_hbm_cache_bytes = 0
    if mode.backend == "latchmoe" or mode.name == "uva-exact":
        if args.plan is not None:
            plan = deserialize_residency_plan(
                args.plan.read_text(encoding="utf-8")
            )
        else:
            config = json.loads((Path(manifest.model.path) / "config.json").read_text())
            config["model_path"] = manifest.model.path
            config["revision"] = manifest.model.revision
            top_k = int(config.get("num_experts_per_tok", 8))
            capture_size = CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE
            if (
                mode.name != "latchmoe-piecewise"
                and manifest.num_slots < capture_size * top_k
            ):
                capture_size = max(1, manifest.num_slots // top_k)
            plan = build_residency_plan(
                args.offload_gib,
                config,
                max_capture_size=capture_size,
                top_k=top_k,
                device_total_bytes=48 << 30,
                kv_reserve_bytes=args.kv_cache_memory_bytes,
            )
        if tuple(plan.offloaded_layer_ids) != tuple(manifest.layer_ids):
            raise RuntimeError(
                "correctness manifest layers do not match midpoint residency plan: "
                f"plan={plan.offloaded_layer_ids}, manifest={manifest.layer_ids}"
            )
        if plan.effective_num_slots != manifest.num_slots:
            raise RuntimeError("correctness manifest slots differ from residency plan")
        if int(plan.ledger["kv_reserve_bytes"]) != args.kv_cache_memory_bytes:
            raise RuntimeError("correctness KV reserve differs from residency plan")
        if plan.model_fingerprint.get("model_path") != manifest.model.path:
            raise RuntimeError("correctness model path differs from residency plan")
        if plan.model_fingerprint.get("revision") != manifest.model.revision:
            raise RuntimeError("correctness model revision differs from residency plan")
        plan_json = serialize_residency_plan(plan)
        backend_hbm_cache_bytes = int(plan.main_slot_cache_bytes)
        identity_lock_json = serialize_identity_lock(
            build_identity_lock(
                plan,
                ["--model", manifest.model.path, "--revision", manifest.model.revision],
            )
        )
        run.write_json("residency_plan.json", plan.to_jsonable())
    run.write_json("environment.json", _environment())
    run.write_json(
        "contract.json",
        {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": manifest.manifest_sha256,
            "plan_path": str(args.plan.resolve()) if args.plan is not None else None,
            "model_path": manifest.model.path,
            "model_revision": manifest.model.revision,
            "vllm_version": manifest.vllm.version,
            "dtype": manifest.dtype,
            "tensor_parallel_size": manifest.tensor_parallel_size,
            "max_num_seqs": args.max_num_seqs,
            "batch_invariant": True,
            "attention_backend": "FLASH_ATTN",
            "max_cudagraph_capture_size": (
                CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE
                if mode.name == "latchmoe-piecewise"
                else None
            ),
            "latchmoe_wave_slots": (
                min(manifest.num_slots, 32)
                if mode.backend == "latchmoe"
                and manifest.num_slots < manifest.model.num_experts
                else None
            ),
            "mode": mode.name,
            "backend": mode.backend,
            "enforce_eager": mode.enforce_eager,
            "expect_waves": mode.expect_waves,
            "overlap_enabled": mode.overlap_enabled,
            "uva_implementation": (
                "ManifestUVAOffloader"
                if mode.name == "uva-exact"
                else ("vllm-stock" if mode.name == "uva" else None)
            ),
            "cpu_offload_gb": (
                STOCK_UVA_CPU_OFFLOAD_GB
                if mode.name in {"uva", "uva-exact"}
                else 0.0
            ),
            "manifest_controls_offload_selection": mode.name != "uva",
            "diagnostic_residual_uva_max_bytes": (
                args.diagnostic_residual_uva_bytes
            ),
            "backend_hbm_cache_bytes": backend_hbm_cache_bytes,
        },
    )
    result_path = run.path / "correctness.json"
    profile_path = run.path / "profile.jsonl"
    worker_command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        args.mode,
        "--manifest",
        str(args.manifest.resolve()),
        "--result-json",
        str(result_path.resolve()),
        "--profile-jsonl",
        str(profile_path.resolve()),
        "--max-tokens",
        str(args.max_tokens),
        "--offload-gib",
        str(args.offload_gib),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--diagnostic-residual-uva-bytes",
        str(args.diagnostic_residual_uva_bytes),
    ]
    if args.prompts_json is not None:
        worker_command.extend(["--prompts-json", str(args.prompts_json.resolve())])
    run.write_json("worker_command.json", worker_command)
    environment = _worker_environment(
        mode,
        args.manifest.resolve(),
        profile_path.resolve(),
        plan_json=plan_json,
        identity_lock_json=identity_lock_json,
        diagnostic_residual_uva_bytes=args.diagnostic_residual_uva_bytes,
        uva_reservation_bytes=(
            backend_hbm_cache_bytes if mode.name == "uva-exact" else 0
        ),
    )
    with (
        (run.path / "stdout.log").open("w", encoding="utf-8") as stdout,
        (run.path / "stderr.log").open("w", encoding="utf-8") as stderr,
    ):
        completed = subprocess.run(
            worker_command,
            cwd=ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise WorkerProcessError(completed.returncode, run.path / "stderr.log")
    if not result_path.is_file():
        raise RuntimeError("correctness worker produced no correctness.json")
    telemetry = _offload_telemetry(mode, manifest, profile_path)
    run.write_json("offload_telemetry.json", telemetry)
    if args.reference_json is not None:
        reference = json.loads(args.reference_json.read_text(encoding="utf-8"))
        candidate = json.loads(result_path.read_text(encoding="utf-8"))
        comparison = compare_greedy_results(reference, candidate)
        reference_telemetry_path = args.reference_json.parent / "offload_telemetry.json"
        if not reference_telemetry_path.is_file():
            raise RuntimeError(
                f"reference has no offload telemetry: {reference_telemetry_path}"
            )
        reference_telemetry = json.loads(
            reference_telemetry_path.read_text(encoding="utf-8")
        )
        offload_bytes_match = (
            reference_telemetry.get("actual_offload_bytes")
            == telemetry["actual_offload_bytes"]
        )
        comparison["reference_actual_offload_bytes"] = reference_telemetry.get(
            "actual_offload_bytes"
        )
        comparison["candidate_actual_offload_bytes"] = telemetry["actual_offload_bytes"]
        comparison["offload_bytes_match"] = offload_bytes_match
        reference_run_manifest = json.loads(
            (args.reference_json.parent / "run_manifest.json").read_text(
                encoding="utf-8"
            )
        )
        candidate_run_manifest = json.loads(
            (run.path / "run_manifest.json").read_text(encoding="utf-8")
        )
        source_state_match = (
            reference_run_manifest.get("source_state_sha256")
            == candidate_run_manifest.get("source_state_sha256")
        )
        comparison["reference_source_state_sha256"] = reference_run_manifest.get(
            "source_state_sha256"
        )
        comparison["candidate_source_state_sha256"] = candidate_run_manifest.get(
            "source_state_sha256"
        )
        comparison["source_state_match"] = source_state_match
        run.write_json("comparison.json", comparison)
        require_greedy_match(reference, candidate)
        if not offload_bytes_match:
            raise CorrectnessMismatchError("actual offload bytes differ")
        if not source_state_match:
            raise CorrectnessMismatchError("source state differs")


def _driver_main(argv: list[str]) -> int:
    args = _parser().parse_args(argv)
    run = ArtifactRun.create(
        args.artifact_dir,
        kind=RunKind(args.kind),
        command=[sys.executable, str(Path(__file__).resolve()), *argv],
    )
    try:
        _execute_driver(args, run)
        run.record_completion(exit_code=0)
        return_code = 0
    except BaseException as error:
        exit_code = getattr(error, "exit_code", 1) or 1
        run.record_failure(error, exit_code=exit_code)
        return_code = exit_code
    finally:
        run.write_inventory()
    return return_code


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--worker" in argv:
        return _worker_main(argv)
    return _driver_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
