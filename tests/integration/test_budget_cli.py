import hashlib
import json

import pytest

from vllm_latchmoe_cuda.cli import (
    CudaMoeCliArgs,
    build_plan_from_model_argument,
    parse_plugin_args,
)
from vllm_latchmoe_cuda.residency_plan import (
    deserialize_residency_plan,
    serialize_residency_plan,
)


def _model(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    config = {
        "model_type": "qwen3_moe",
        "hidden_size": 2,
        "moe_intermediate_size": 2,
        "num_experts": 4,
        "num_experts_per_tok": 1,
        "num_hidden_layers": 4,
        "torch_dtype": "bfloat16",
    }
    (root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {}}), encoding="utf-8"
    )
    return root


def test_parser_removes_only_cuda_budget_flag():
    plugin_args, vllm_args = parse_plugin_args(
        ["serve", "--model", "/model", "--cuda-moe-offload-gb", "13.5", "--port", "8000"]
    )
    assert plugin_args == CudaMoeCliArgs(offload_gib=13.5)
    assert vllm_args == ["serve", "--model", "/model", "--port", "8000"]


@pytest.mark.parametrize(
    "environment,extra",
    [
        ({}, ["--cpu-offload-gb", "1"]),
        ({"VLLM_LATCHMOE_MANIFEST": "old.json"}, []),
        ({"VLLM_LATCHMOE_WAVE_SLOTS": "8"}, []),
    ],
)
def test_parser_rejects_competing_production_controls(monkeypatch, environment, extra):
    for key in ("VLLM_LATCHMOE_MANIFEST", "VLLM_LATCHMOE_WAVE_SLOTS"):
        monkeypatch.delenv(key, raising=False)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    with pytest.raises(ValueError):
        parse_plugin_args(["--cuda-moe-offload-gb", "1", *extra])


def test_parent_and_worker_round_trip_exact_same_plan(monkeypatch, tmp_path):
    root = _model(tmp_path)
    monkeypatch.setenv("VLLM_LATCHMOE_DEVICE_TOTAL_BYTES", str(1 << 30))
    parent = build_plan_from_model_argument(
        CudaMoeCliArgs(offload_gib=47 / (1 << 30)),
        ["serve", "--model", str(root), "--max-cudagraph-capture-size", "1"],
    )
    worker = deserialize_residency_plan(serialize_residency_plan(parent))

    assert worker.plan_id == parent.plan_id
    assert worker.selection_strategy == parent.selection_strategy
    assert worker.eligible_layer_ids == parent.eligible_layer_ids
    assert worker.offloaded_layer_ids == parent.offloaded_layer_ids
    assert parent.model_fingerprint["config_sha256"] == hashlib.sha256(
        (root / "config.json").read_bytes()
    ).hexdigest()
