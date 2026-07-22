#!/usr/bin/env python3
"""Run resident Qwen3-30B INT8 eager/graph CUDA-timeline experiments."""

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

from analyze_cuda_timeline import analyze_trace


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path(sys.executable)
VLLM = PYTHON.with_name("vllm")
DATASET_NAMES = ("ShareGPT", "LongBench", "HumanEval", "GSM8K")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("/home/lcw/model"))
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--num-prompts", type=int, default=50)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--max-concurrency", type=int, default=8)
    parser.add_argument("--max-num-seqs", type=int, default=8)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-num-batched-tokens", type=int, default=512)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=268435456)
    parser.add_argument("--profile-delay-iterations", type=int, default=16)
    parser.add_argument("--profile-iterations", type=int, default=80)
    parser.add_argument("--port", type=int, default=8128)
    parser.add_argument("--startup-timeout-s", type=float, default=600)
    return parser.parse_args()


def clean_env() -> dict[str, str]:
    env = os.environ.copy()
    for key in (
        "VLLM_PLUGINS",
        "VLLM_LATCHMOE_MODE",
        "VLLM_LATCHMOE_MANIFEST",
        "VLLM_LATCHMOE_PROFILE_PATH",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "ALL_PROXY",
        "http_proxy",
        "https_proxy",
        "all_proxy",
    ):
        env.pop(key, None)
    env.update(
        {
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
            "HF_HUB_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "VLLM_DISABLE_COMPILE_CACHE": "1",
        }
    )
    return env


def gpu_snapshot() -> dict[str, object]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.used,memory.free,temperature.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    name, total, used, free, temperature = [
        item.strip() for item in result.stdout.splitlines()[0].split(",")
    ]
    return {
        "name": name,
        "memory_total_mib": int(total),
        "memory_used_mib": int(used),
        "memory_free_mib": int(free),
        "temperature_c": int(temperature),
    }


def opener() -> urllib.request.OpenerDirector:
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def wait_for_server(url: str, process: subprocess.Popen[str], timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"server exited with code {process.returncode}")
        try:
            with opener().open(url, timeout=5):
                return
        except (urllib.error.URLError, TimeoutError, OSError):
            time.sleep(2)
    raise TimeoutError(f"server did not become ready: {url}")


def post(url: str) -> None:
    request = urllib.request.Request(url, data=b"", method="POST")
    with opener().open(request, timeout=180) as response:
        if response.status >= 300:
            raise RuntimeError(f"POST {url} returned {response.status}")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def server_command(
    args: argparse.Namespace, mode: str, profile_dir: Path
) -> list[str]:
    graph_mode = "NONE" if mode == "eager" else "PIECEWISE"
    profile_config = {
        "profiler": "torch",
        "torch_profiler_dir": str(profile_dir.resolve()),
        "torch_profiler_with_stack": False,
        "torch_profiler_use_gzip": False,
        "torch_profiler_dump_cuda_time_total": False,
        "ignore_frontend": True,
        "delay_iterations": args.profile_delay_iterations,
        "max_iterations": args.profile_iterations,
    }
    return [
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
        "--compilation-config",
        json.dumps({"cudagraph_mode": graph_mode}),
        "--profiler-config",
        json.dumps(profile_config),
    ]


def client_command(
    args: argparse.Namespace,
    dataset: Path,
    mode: str,
    dataset_name: str,
    output_dir: Path,
    *,
    warmup: bool,
) -> list[str]:
    prompt_count = args.max_concurrency if warmup else args.num_prompts
    output_len = 8 if warmup else args.output_len
    command = [
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
        "custom",
        "--dataset-path",
        str(dataset),
        "--custom-output-len",
        str(output_len),
        "--num-prompts",
        str(prompt_count),
        "--request-rate",
        "inf",
        "--max-concurrency",
        str(args.max_concurrency),
        "--seed",
        "42",
        "--num-warmups",
        "0",
        "--temperature",
        "0",
        "--ignore-eos",
        "--no-oversample",
        "--disable-tqdm",
        "--request-id-prefix",
        f"prof-{dataset_name.lower()}-{mode}-{'warmup' if warmup else 'run'}-",
    ]
    if not warmup:
        command.extend(
            [
                "--percentile-metrics",
                "ttft,tpot,itl,e2el",
                "--metric-percentiles",
                "50,90,99",
                "--save-result",
                "--save-detailed",
                "--result-dir",
                str(output_dir),
                "--result-filename",
                "client_result.json",
            ]
        )
    return command


def run_client(command: list[str], env: dict[str, str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if completed.returncode:
        raise RuntimeError(f"client failed with code {completed.returncode}: {command}")


def wait_for_trace(profile_dir: Path, timeout: float = 120) -> Path:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        traces = sorted(profile_dir.glob("*.pt.trace.json"))
        if traces and traces[-1].stat().st_size > 0:
            return traces[-1]
        time.sleep(1)
    raise TimeoutError(f"profiler trace did not appear in {profile_dir}")


def run_case(
    args: argparse.Namespace, dataset_name: str, dataset: Path, mode: str
) -> dict[str, object]:
    case_dir = args.artifact_dir / dataset_name.lower() / mode
    profile_dir = case_dir / "cuda-timeline"
    profile_dir.mkdir(parents=True)
    env = clean_env()
    command = server_command(args, mode, profile_dir)
    (case_dir / "server_command.json").write_text(
        json.dumps(command, indent=2) + "\n", encoding="utf-8"
    )
    with (case_dir / "server.log").open("w", encoding="utf-8") as server_log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        try:
            base_url = f"http://127.0.0.1:{args.port}"
            wait_for_server(base_url + "/v1/models", process, args.startup_timeout_s)
            warmup = client_command(
                args, dataset, mode, dataset_name, case_dir, warmup=True
            )
            run_client(warmup, env, case_dir / "warmup.log")
            post(base_url + "/start_profile")
            client = client_command(
                args, dataset, mode, dataset_name, case_dir, warmup=False
            )
            (case_dir / "client_command.json").write_text(
                json.dumps(client, indent=2) + "\n", encoding="utf-8"
            )
            run_client(client, env, case_dir / "client.log")
            post(base_url + "/stop_profile")
            trace_path = wait_for_trace(profile_dir)
            client_result = json.loads(
                (case_dir / "client_result.json").read_text(encoding="utf-8")
            )
            if client_result.get("completed") != args.num_prompts:
                raise RuntimeError(
                    f"{dataset_name}/{mode}: only {client_result.get('completed')} "
                    f"of {args.num_prompts} requests completed"
                )
            average_tpot_ms = float(client_result["mean_tpot_ms"])
            profile = analyze_trace(trace_path, average_tpot_ms)
            (case_dir / "timeline_summary.json").write_text(
                json.dumps(profile, indent=2) + "\n", encoding="utf-8"
            )
            return {
                "dataset": dataset_name,
                "mode": mode,
                "gpu_after": gpu_snapshot(),
                "client": {
                    key: client_result.get(key)
                    for key in (
                        "completed",
                        "failed",
                        "total_input_tokens",
                        "total_output_tokens",
                        "duration",
                        "mean_tpot_ms",
                        "median_tpot_ms",
                        "output_throughput",
                    )
                },
                "timeline": {
                    key: profile[key]
                    for key in (
                        "trace",
                        "definition",
                        "decode_iterations",
                        "generation_batch_sizes",
                        "average_tpot_ms",
                        "metrics",
                    )
                },
            }
        finally:
            stop(process)


def main() -> None:
    args = parse_args()
    if args.artifact_dir.exists():
        raise SystemExit(f"artifact directory already exists: {args.artifact_dir}")
    if not args.model.is_dir():
        raise SystemExit(f"model does not exist: {args.model}")
    datasets = {
        name: args.dataset_dir / f"{name.lower()}.jsonl" for name in DATASET_NAMES
    }
    missing = [str(path) for path in datasets.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing prepared datasets: {missing}")
    args.artifact_dir.mkdir(parents=True)
    contract = {
        "model": str(args.model.resolve()),
        "quantization": "experts_int8",
        "dtype": "bfloat16",
        "cpu_offload_gb": 0,
        "only_runtime_variable_within_each_dataset": "cudagraph_mode NONE vs PIECEWISE",
        "datasets": {name: str(path.resolve()) for name, path in datasets.items()},
        "num_prompts": args.num_prompts,
        "output_len": args.output_len,
        "max_concurrency": args.max_concurrency,
        "max_num_seqs": args.max_num_seqs,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "kv_cache_memory_bytes": args.kv_cache_memory_bytes,
        "profile_delay_iterations": args.profile_delay_iterations,
        "profile_iterations": args.profile_iterations,
        "timeline_decomposition": {
            "device_execution": (
                "union(kernel); excludes gpu_memcpy and gpu_memset"
            ),
            "host_induced_device_gaps": (
                "mean decode TPOT - kernel-only device execution"
            ),
            "average_tpot": (
                "client mean_tpot_ms; both components cover decode only and exclude "
                "prefill/TTFT"
            ),
            "unit": "milliseconds per decode token",
        },
    }
    (args.artifact_dir / "contract.json").write_text(
        json.dumps(contract, indent=2) + "\n", encoding="utf-8"
    )
    (args.artifact_dir / "gpu_before.json").write_text(
        json.dumps(gpu_snapshot(), indent=2) + "\n", encoding="utf-8"
    )
    results = {name: {} for name in DATASET_NAMES}
    for name, dataset in datasets.items():
        for mode in ("eager", "graph"):
            print(f"running {name}/{mode}", flush=True)
            results[name][mode] = run_case(args, name, dataset, mode)
            (args.artifact_dir / "results.partial.json").write_text(
                json.dumps(results, indent=2) + "\n", encoding="utf-8"
            )
    (args.artifact_dir / "results.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
