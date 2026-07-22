#!/usr/bin/env python3
"""Run a resident INT8 Qwen3 eager-vs-PIECEWISE ablation.

The only runtime variable between the two arms is CUDA Graph replay.  Qwen3-MoE
uses vLLM's ``experts_int8`` path: expert matrices are quantized during load,
while the small dense/router part remains in the requested BF16 compute dtype.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
VLLM = PYTHON.with_name("vllm")
MODEL = Path("/home/lcw/model")
DATASET = Path("/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json")


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=MODEL)
    p.add_argument("--dataset", type=Path, default=DATASET)
    p.add_argument("--artifact-dir", type=Path, required=True)
    p.add_argument("--repetitions", type=int, default=3)
    p.add_argument("--num-prompts", type=int, default=50)
    p.add_argument("--output-len", type=int, default=128)
    p.add_argument("--max-concurrency", type=int, default=8)
    p.add_argument("--max-num-seqs", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--max-num-batched-tokens", type=int, default=512)
    p.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    p.add_argument("--port", type=int, default=8126)
    p.add_argument("--startup-timeout-s", type=float, default=600)
    return p.parse_args()


def _gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    name, total, used, free = [
        part.strip() for part in result.stdout.splitlines()[0].split(",")
    ]
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "memory_free_mib": int(free),
    }


def _wait_for_server(url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with opener.open(url, timeout=5):
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {url}")


def _stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _server_command(args: argparse.Namespace, mode: str) -> list[str]:
    command = [
        str(VLLM),
        "serve",
        str(args.model),
        "--served-model-name",
        "qwen3-30b-int8",
        "--quantization",
        "experts_int8",
        "--allow-deprecated-quantization",
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
        "--gpu-memory-utilization",
        "0.95",
        "--kv-cache-memory-bytes",
        str(args.kv_cache_memory_bytes),
        "--no-enable-prefix-caching",
        "--disable-log-stats",
        "--generation-config",
        "vllm",
    ]
    # Keep vLLM's compile path identical in both arms.  NONE is the eager
    # control; --enforce-eager would also disable torch.compile.
    graph_mode = "NONE" if mode == "eager" else "PIECEWISE"
    command.extend(["--compilation-config", json.dumps({"cudagraph_mode": graph_mode})])
    return command


def _client_command(
    args: argparse.Namespace, mode: str, repetition: int, out_dir: Path
) -> list[str]:
    return [
        str(VLLM),
        "bench",
        "serve",
        "--backend",
        "openai",
        "--base-url",
        f"http://127.0.0.1:{args.port}",
        "--endpoint",
        "/v1/completions",
        "--model",
        "qwen3-30b-int8",
        "--tokenizer",
        str(args.model),
        "--dataset-name",
        "sharegpt",
        "--dataset-path",
        str(args.dataset),
        "--num-prompts",
        str(args.num_prompts),
        "--sharegpt-output-len",
        str(args.output_len),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.max_concurrency),
        "--seed",
        "42",
        "--num-warmups",
        "2",
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
        f"int8-{mode}-{repetition}-",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(out_dir),
        "--result-filename",
        "raw_result.json",
    ]


def _run_mode(args: argparse.Namespace, root: Path, mode: str) -> dict[str, object]:
    mode_dir = root / mode
    mode_dir.mkdir()
    log_path = mode_dir / "server.log"
    command = _server_command(args, mode)
    (mode_dir / "server_command.json").write_text(json.dumps(command, indent=2) + "\n")
    env = os.environ.copy()
    for key in (
        "VLLM_PLUGINS",
        "VLLM_LATCHMOE_MODE",
        "VLLM_LATCHMOE_MANIFEST",
        "VLLM_LATCHMOE_PROFILE_PATH",
    ):
        env.pop(key, None)
    for key in (
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env["NO_PROXY"] = "127.0.0.1,localhost"
    env["no_proxy"] = "127.0.0.1,localhost"
    env.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            _wait_for_server(
                f"http://127.0.0.1:{args.port}/v1/models",
                process,
                args.startup_timeout_s,
            )
            measurements = []
            for repetition in range(1, args.repetitions + 1):
                rep_dir = mode_dir / f"repetition-{repetition:02d}"
                rep_dir.mkdir()
                client = _client_command(args, mode, repetition, rep_dir)
                (rep_dir / "client_command.json").write_text(
                    json.dumps(client, indent=2) + "\n"
                )
                with (rep_dir / "client.log").open("w", encoding="utf-8") as client_log:
                    completed = subprocess.run(
                        client,
                        cwd=ROOT,
                        env=env,
                        stdout=client_log,
                        stderr=subprocess.STDOUT,
                        text=True,
                    )
                if completed.returncode:
                    raise RuntimeError(
                        f"{mode} repetition {repetition} failed with {completed.returncode}"
                    )
                raw = json.loads((rep_dir / "raw_result.json").read_text())
                if (
                    raw.get("completed") != args.num_prompts
                    or raw.get("failed", 0) != 0
                ):
                    raise RuntimeError(
                        f"incomplete {mode} repetition {repetition}: {raw.get('completed')} completed, {raw.get('failed')} failed"
                    )
                measurement = {
                    "completed": raw["completed"],
                    "failed": raw["failed"],
                    "total_input_tokens": raw["total_input_tokens"],
                    "total_output_tokens": raw["total_output_tokens"],
                    "duration_s": raw["duration"],
                    "metrics": {
                        key: raw[key]
                        for key in (
                            "median_ttft_ms",
                            "mean_ttft_ms",
                            "p99_ttft_ms",
                            "median_tpot_ms",
                            "mean_tpot_ms",
                            "p99_tpot_ms",
                            "output_throughput",
                            "request_throughput",
                        )
                    },
                }
                (rep_dir / "measurement.json").write_text(
                    json.dumps(measurement, indent=2) + "\n"
                )
                measurements.append(measurement)
            summary = {
                "mode": mode,
                "repetitions": measurements,
                "gpu_after": _gpu_snapshot(),
            }
            for metric in measurements[0]["metrics"]:
                values = [float(item["metrics"][metric]) for item in measurements]
                summary.setdefault("metrics", {})[metric] = {
                    "values": values,
                    "mean": sum(values) / len(values),
                    "median": sorted(values)[len(values) // 2],
                }
            (mode_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
            return summary
        finally:
            _stop(process)


def main() -> None:
    args = _args()
    if args.repetitions < 3:
        raise SystemExit("use at least three repetitions for the final result")
    if not args.model.is_dir() or not args.dataset.is_file():
        raise SystemExit("model and dataset paths must exist")
    if args.artifact_dir.exists():
        raise SystemExit(f"artifact directory already exists: {args.artifact_dir}")
    args.artifact_dir.mkdir(parents=True)
    contract = {
        "model": str(args.model.resolve()),
        "quantization": "experts_int8",
        "dtype": "bfloat16",
        "cpu_offload_gb": 0,
        "graph_variable": "eager vs PIECEWISE",
        "dataset": str(args.dataset.resolve()),
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "max_concurrency": args.max_concurrency,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
    }
    (args.artifact_dir / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    (args.artifact_dir / "gpu_before.json").write_text(
        json.dumps(_gpu_snapshot(), indent=2) + "\n"
    )
    results = {
        mode: _run_mode(args, args.artifact_dir, mode) for mode in ("eager", "graph")
    }
    (args.artifact_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
