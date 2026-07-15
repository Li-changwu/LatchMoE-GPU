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
        choices=("uva", "latchmoe-eager", "latchmoe-piecewise", "latchmoe-waves"),
    )
    parser.add_argument(
        "--kind", choices=tuple(kind.value for kind in RunKind), required=True
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--prompts-json", type=Path)
    parser.add_argument("--reference-json", type=Path)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.98)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=256 * 1024 * 1024)
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
    parser.add_argument("--max-model-len", type=int, required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, required=True)
    parser.add_argument("--kv-cache-memory-bytes", type=int, required=True)
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
    if mode.name == "uva":
        matches = [event for event in events if event.get("event") == "stock_uva"]
        if len(matches) != 1:
            raise RuntimeError("stock UVA run produced no unique stock_uva event")
        event = matches[0]
        implementation = event.get("implementation")
        if implementation != "vllm.model_executor.offloader.uva.UVAOffloader":
            raise RuntimeError(f"unexpected UVA implementation: {implementation}")
        return {
            "schema_version": 1,
            "mode": mode.name,
            "implementation": implementation,
            "manifest_bytes": 0,
            "residual_uva_bytes": int(event["cpu_offload_bytes"]),
            "actual_offload_bytes": int(event["cpu_offload_bytes"]),
            "configured_budget_bytes": int(event["cpu_offload_max_bytes"]),
        }
    matches = [event for event in events if event.get("event") == "residual_uva"]
    if len(matches) != 1:
        raise RuntimeError("LatchMoE run produced no unique residual_uva event")
    event = matches[0]
    manifest_bytes = manifest.total_elements * 2
    residual_bytes = int(event["cpu_offload_bytes"])
    return {
        "schema_version": 1,
        "mode": mode.name,
        "implementation": "vllm_latchmoe_cuda.offloader.CudaSEWOffloader",
        "residual_implementation": "vllm.model_executor.offloader.uva.UVAOffloader",
        "manifest_bytes": manifest_bytes,
        "residual_uva_bytes": residual_bytes,
        "actual_offload_bytes": manifest_bytes + residual_bytes,
        "configured_budget_bytes": (
            manifest_bytes + int(event["cpu_offload_max_bytes"])
        ),
    }


def _worker_environment(
    mode: CorrectnessMode, manifest_path: Path, profile_path: Path
) -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "VLLM_LATCHMOE_MODE",
        "VLLM_LATCHMOE_MANIFEST",
        "VLLM_LATCHMOE_PROFILE_PATH",
        "VLLM_LATCHMOE_TELEMETRY_PATH",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "VLLM_PLUGINS": "latchmoe_cuda",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if mode.backend == "latchmoe":
        environment.update(
            {
                "VLLM_LATCHMOE_MODE": mode.backend,
                "VLLM_LATCHMOE_MANIFEST": str(manifest_path),
                "VLLM_LATCHMOE_PROFILE_PATH": str(profile_path),
            }
        )
    else:
        environment["VLLM_LATCHMOE_TELEMETRY_PATH"] = str(profile_path)
    return environment


def _execute_driver(args, run: ArtifactRun) -> None:
    manifest = OffloadManifest.load(args.manifest)
    manifest.validate_model_files()
    mode = resolve_correctness_mode(args.mode)
    run.write_json("environment.json", _environment())
    run.write_json(
        "contract.json",
        {
            "manifest_path": str(args.manifest.resolve()),
            "manifest_sha256": manifest.manifest_sha256,
            "model_path": manifest.model.path,
            "model_revision": manifest.model.revision,
            "vllm_version": manifest.vllm.version,
            "dtype": manifest.dtype,
            "tensor_parallel_size": manifest.tensor_parallel_size,
            "mode": mode.name,
            "backend": mode.backend,
            "enforce_eager": mode.enforce_eager,
            "expect_waves": mode.expect_waves,
            "uva_implementation": "vllm-stock" if mode.name == "uva" else None,
            "cpu_offload_gb": (STOCK_UVA_CPU_OFFLOAD_GB if mode.name == "uva" else 0.0),
            "manifest_controls_offload_selection": mode.name != "uva",
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
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
    ]
    if args.prompts_json is not None:
        worker_command.extend(["--prompts-json", str(args.prompts_json.resolve())])
    run.write_json("worker_command.json", worker_command)
    environment = _worker_environment(
        mode,
        args.manifest.resolve(),
        profile_path.resolve(),
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
        run.write_json("comparison.json", comparison)
        require_greedy_match(reference, candidate)
        if not offload_bytes_match:
            raise CorrectnessMismatchError("actual offload bytes differ")


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
