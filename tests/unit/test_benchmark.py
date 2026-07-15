import json
from dataclasses import replace

import pytest

from vllm_latchmoe_cuda.benchmark import (
    build_client_command,
    build_server_command,
    compare_ablation_summaries,
    compare_mode_summaries,
    local_benchmark_environment,
    normalize_benchmark_result,
    read_offload_telemetry,
    summarize_repetitions,
)


def _raw_result(value: float = 1.0):
    return {
        "completed": 50,
        "failed": 0,
        "total_input_tokens": 5000,
        "total_output_tokens": 6400,
        "duration": 100.0,
        "median_ttft_ms": 100 * value,
        "mean_ttft_ms": 110 * value,
        "p99_ttft_ms": 200 * value,
        "median_tpot_ms": 20 * value,
        "mean_tpot_ms": 22 * value,
        "p99_tpot_ms": 30 * value,
        "output_throughput": 64 / value,
        "request_throughput": 0.5 / value,
    }


def test_commands_lock_official_uva_and_identical_sharegpt_workload(tiny_manifest):
    server = build_server_command(
        python_executable="/env/bin/python",
        mode="uva",
        manifest=tiny_manifest,
        host="127.0.0.1",
        port=8026,
        served_model_name="qwen",
        max_num_seqs=8,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=268435456,
    )
    client = build_client_command(
        python_executable="/env/bin/python",
        base_url="http://127.0.0.1:8026",
        served_model_name="qwen",
        tokenizer="/model",
        dataset_path="/data/sharegpt.json",
        result_dir="/result",
        result_filename="raw.json",
        num_prompts=50,
        output_len=128,
        max_concurrency=8,
        seed=42,
        warmup_requests=2,
        request_id_prefix="rep-1-",
    )

    assert server[0] == "/env/bin/vllm"
    assert server[server.index("--cpu-offload-gb") + 1] == "14.0"
    assert "--enforce-eager" in server
    assert "--disable-log-stats" in server
    assert "--no-enable-prefix-caching" in server
    assert client[client.index("--dataset-name") + 1] == "sharegpt"
    assert client[client.index("--num-prompts") + 1] == "50"
    assert client[client.index("--sharegpt-output-len") + 1] == "128"
    assert client[client.index("--max-concurrency") + 1] == "8"
    assert "--ignore-eos" in client


def test_latchmoe_server_does_not_enable_stock_offload(tiny_manifest):
    command = build_server_command(
        python_executable="/env/bin/python",
        mode="latchmoe-eager",
        manifest=tiny_manifest,
        host="127.0.0.1",
        port=8026,
        served_model_name="qwen",
        max_num_seqs=1,
        max_model_len=512,
        max_num_batched_tokens=512,
        kv_cache_memory_bytes=268435456,
    )

    assert "--cpu-offload-gb" not in command
    assert "--enforce-eager" in command


@pytest.mark.parametrize("mode", ["uva-piecewise", "latchmoe-piecewise"])
def test_piecewise_server_modes_use_identical_graph_policy(mode, tiny_manifest):
    command = build_server_command(
        python_executable="/env/bin/python",
        mode=mode,
        manifest=tiny_manifest,
        host="127.0.0.1",
        port=8026,
        served_model_name="qwen",
        max_num_seqs=8,
        max_model_len=2048,
        max_num_batched_tokens=2048,
        kv_cache_memory_bytes=268435456,
    )

    assert "--enforce-eager" not in command
    config = json.loads(command[command.index("--compilation-config") + 1])
    assert config["cudagraph_mode"] == "PIECEWISE"
    assert config["custom_ops"] == ["+unquantized_fused_moe"]
    assert command[command.index("--max-cudagraph-capture-size") + 1] == "8"
    assert "--disable-log-stats" in command
    assert "--cudagraph-metrics" not in command
    assert ("--cpu-offload-gb" in command) is mode.startswith("uva")


def test_local_benchmark_environment_bypasses_proxies():
    environment = local_benchmark_environment(
        {
            "HTTP_PROXY": "socks5://proxy:1080",
            "https_proxy": "socks5://proxy:1080",
            "PATH": "/bin",
        }
    )

    assert "HTTP_PROXY" not in environment
    assert "https_proxy" not in environment
    assert environment["NO_PROXY"] == "127.0.0.1,localhost"
    assert environment["no_proxy"] == "127.0.0.1,localhost"
    assert environment["PATH"] == "/bin"


def test_normalize_and_summarize_require_three_complete_fixed_length_runs():
    normalized = [
        normalize_benchmark_result(
            _raw_result(value), expected_requests=50, expected_output_len=128
        )
        for value in (1.0, 1.1, 0.9)
    ]

    summary = summarize_repetitions(normalized)

    assert summary["median_ttft_ms"]["median"] == pytest.approx(100.0)
    assert summary["output_throughput"]["median"] == pytest.approx(64.0)
    with pytest.raises(ValueError, match="at least 3"):
        summarize_repetitions(normalized[:2])


def test_normalize_rejects_early_eos():
    raw = _raw_result()
    raw["total_output_tokens"] = 6399

    with pytest.raises(RuntimeError, match="output token count"):
        normalize_benchmark_result(raw, expected_requests=50, expected_output_len=128)


def test_telemetry_proves_offload_bytes_and_device_planner(tiny_manifest, tmp_path):
    profile = tmp_path / "profile.jsonl"
    profile.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                {
                    "event": "residual_uva",
                    "cpu_offload_bytes": 64,
                    "cpu_offload_max_bytes": 64,
                },
                {
                    "event": "exact_waves",
                    "pair_planner_mode": "cuda_device",
                    "scatter_mode": "layer_index_add",
                },
            )
        )
        + "\n",
        encoding="utf-8",
    )

    telemetry = read_offload_telemetry("latchmoe-eager", tiny_manifest, profile)

    assert telemetry["actual_offload_bytes"] == tiny_manifest.total_elements * 2 + 64
    assert telemetry["exact_wave_events"] == 1
    assert telemetry["cuda_device_planner_events"] == 1


def test_full_capacity_eager_telemetry_requires_every_direct_slot_layer(
    tiny_manifest, tmp_path
):
    full_capacity_manifest = replace(tiny_manifest, num_slots=4)
    profile = tmp_path / "profile.jsonl"
    backend = {
        "event": "residual_uva",
        "cpu_offload_bytes": 64,
        "cpu_offload_max_bytes": 64,
    }
    direct = {
        "event": "direct_slots",
        "layer_id": 0,
        "active_experts": 4,
        "slot_capacity": 4,
    }
    profile.write_text(
        "\n".join(json.dumps(event) for event in (backend, direct)) + "\n",
        encoding="utf-8",
    )

    telemetry = read_offload_telemetry(
        "latchmoe-eager", full_capacity_manifest, profile
    )

    assert telemetry["direct_slot_events"] == 1
    assert telemetry["exact_wave_events"] == 0
    profile.write_text(json.dumps(backend) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="every full-capacity direct-slot layer"):
        read_offload_telemetry("latchmoe-eager", full_capacity_manifest, profile)


@pytest.mark.parametrize("mode", ["uva-piecewise", "latchmoe-piecewise"])
def test_piecewise_telemetry_requires_capture_and_replay(mode, tiny_manifest, tmp_path):
    profile = tmp_path / "profile.jsonl"
    backend_event = (
        {
            "event": "stock_uva",
            "implementation": "vllm.model_executor.offloader.uva.UVAOffloader",
            "cpu_offload_bytes": 1024,
            "cpu_offload_max_bytes": 1024,
        }
        if mode.startswith("uva")
        else {
            "event": "residual_uva",
            "cpu_offload_bytes": 64,
            "cpu_offload_max_bytes": 64,
        }
    )
    profile.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                backend_event,
                {"event": "cudagraph_capture", "runtime_mode": "PIECEWISE"},
                {"event": "cudagraph_replay", "runtime_mode": "PIECEWISE"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    telemetry = read_offload_telemetry(mode, tiny_manifest, profile)

    assert telemetry["cudagraph_captures"] == 1
    assert telemetry["cudagraph_replays"] == 1
    profile.write_text(
        "\n".join(
            json.dumps(event)
            for event in (
                backend_event,
                {"event": "cudagraph_capture", "runtime_mode": "FULL"},
                {"event": "cudagraph_replay", "runtime_mode": "FULL"},
            )
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="capture and replay"):
        read_offload_telemetry(mode, tiny_manifest, profile)
    profile.write_text(json.dumps(backend_event) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="capture and replay"):
        read_offload_telemetry(mode, tiny_manifest, profile)


def test_comparison_uses_latency_reduction_and_throughput_gain():
    metrics = summarize_repetitions(
        [
            normalize_benchmark_result(
                _raw_result(), expected_requests=50, expected_output_len=128
            )
            for _ in range(3)
        ]
    )
    faster = summarize_repetitions(
        [
            normalize_benchmark_result(
                _raw_result(0.5), expected_requests=50, expected_output_len=128
            )
            for _ in range(3)
        ]
    )
    common = {
        "workload_contract_sha256": "a" * 64,
        "offload_telemetry": {"actual_offload_bytes": 1024},
    }

    result = compare_mode_summaries(
        {**common, "mode": "uva", "metrics": metrics},
        {**common, "mode": "latchmoe-eager", "metrics": faster},
    )

    assert result["metrics"]["median_ttft_ms"]["improvement_percent"] == 50
    assert result["metrics"]["output_throughput"]["improvement_percent"] == 100


def test_piecewise_comparison_rejects_mixed_graph_policy():
    metrics = summarize_repetitions(
        [
            normalize_benchmark_result(
                _raw_result(), expected_requests=50, expected_output_len=128
            )
            for _ in range(3)
        ]
    )
    common = {
        "workload_contract_sha256": "a" * 64,
        "offload_telemetry": {"actual_offload_bytes": 1024},
        "metrics": metrics,
    }

    with pytest.raises(ValueError, match="graph policy"):
        compare_mode_summaries(
            {**common, "mode": "uva"},
            {**common, "mode": "latchmoe-piecewise"},
        )

    result = compare_mode_summaries(
        {**common, "mode": "uva-piecewise"},
        {**common, "mode": "latchmoe-piecewise"},
    )
    assert result["reference_mode"] == "uva-piecewise"


def test_ablation_comparison_requires_one_contract_source_and_offload_budget():
    metrics = summarize_repetitions(
        [
            normalize_benchmark_result(
                _raw_result(), expected_requests=50, expected_output_len=128
            )
            for _ in range(3)
        ]
    )
    faster = summarize_repetitions(
        [
            normalize_benchmark_result(
                _raw_result(0.5), expected_requests=50, expected_output_len=128
            )
            for _ in range(3)
        ]
    )
    common = {
        "workload_contract_sha256": "a" * 64,
        "source_state_sha256": "b" * 64,
        "offload_telemetry": {"actual_offload_bytes": 1024},
    }
    summaries = {
        "uva_eager": {**common, "mode": "uva", "metrics": metrics},
        "uva_piecewise": {
            **common,
            "mode": "uva-piecewise",
            "metrics": faster,
        },
        "latchmoe_eager": {
            **common,
            "mode": "latchmoe-eager",
            "metrics": faster,
        },
        "latchmoe_piecewise": {
            **common,
            "mode": "latchmoe-piecewise",
            "metrics": faster,
        },
    }

    result = compare_ablation_summaries(**summaries)

    assert (
        result["graph_effect"]["uva"]["metrics"]["median_tpot_ms"][
            "improvement_percent"
        ]
        == 50
    )
    assert (
        result["latchmoe_effect"]["piecewise"]["metrics"]["output_throughput"][
            "improvement_percent"
        ]
        == 0
    )
    summaries["latchmoe_piecewise"] = {
        **summaries["latchmoe_piecewise"],
        "source_state_sha256": "c" * 64,
    }
    with pytest.raises(ValueError, match="source states differ"):
        compare_ablation_summaries(**summaries)
