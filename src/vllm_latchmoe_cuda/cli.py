"""Production command-line wrapper for the CUDA LatchMoE plugin."""

from __future__ import annotations

import json
import hashlib
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch

from .manifest import build_identity_lock, serialize_identity_lock
from .residency_plan import (
    build_residency_plan,
    serialize_residency_plan,
)


@dataclass(frozen=True)
class CudaMoeCliArgs:
    offload_gib: float
    diagnostic_mode: bool = False


def parse_plugin_args(argv: Sequence[str]) -> tuple[CudaMoeCliArgs, list[str]]:
    """Extract plugin flags while leaving the remaining vLLM argv untouched."""
    remaining: list[str] = []
    offload: float | None = None
    diagnostic = False
    index = 0
    values = list(argv)
    while index < len(values):
        value = values[index]
        if value in {"--cuda-moe-diagnostic", "--cuda-moe-diagnostic-mode"}:
            diagnostic = True
            index += 1
            continue
        if value == "--cuda-moe-offload-gb":
            if index + 1 >= len(values):
                raise ValueError("--cuda-moe-offload-gb requires a value")
            raw = values[index + 1]
            index += 2
        elif value.startswith("--cuda-moe-offload-gb="):
            raw = value.split("=", 1)[1]
            index += 1
        else:
            remaining.append(value)
            index += 1
            continue
        if offload is not None:
            raise ValueError("--cuda-moe-offload-gb may only be provided once")
        try:
            offload = float(raw)
        except ValueError as exc:
            raise ValueError("--cuda-moe-offload-gb must be numeric") from exc
        if offload < 0:
            raise ValueError("--cuda-moe-offload-gb must be non-negative")
    if offload is None:
        raise ValueError("production mode requires --cuda-moe-offload-gb")
    if not diagnostic:
        if any(value == "--cpu-offload-gb" or value.startswith("--cpu-offload-gb=") for value in remaining):
            raise ValueError("--cpu-offload-gb cannot be combined with CUDA residency")
        if os.getenv("VLLM_LATCHMOE_MANIFEST"):
            raise ValueError("VLLM_LATCHMOE_MANIFEST is diagnostic-only")
        if os.getenv("VLLM_LATCHMOE_WAVE_SLOTS"):
            raise ValueError("VLLM_LATCHMOE_WAVE_SLOTS is diagnostic-only")
    return CudaMoeCliArgs(offload_gib=offload, diagnostic_mode=diagnostic), remaining


def _arg_value(argv: Sequence[str], name: str) -> str | None:
    values = list(argv)
    for index, value in enumerate(values):
        if value == name and index + 1 < len(values):
            return values[index + 1]
        if value.startswith(name + "="):
            return value.split("=", 1)[1]
    return None


def _device_total_bytes() -> int:
    override = os.getenv("VLLM_LATCHMOE_DEVICE_TOTAL_BYTES")
    if override:
        return int(override)
    if torch.cuda.is_available():
        return int(torch.cuda.get_device_properties(0).total_memory)
    # A CPU-only parent can still produce a plan for a remote worker.  The
    # worker's explicit device check remains the final feasibility gate.
    return 1 << 60


def build_plan_from_model_argument(
    plugin_args: CudaMoeCliArgs, vllm_args: Sequence[str]
):
    model_arg = _arg_value(vllm_args, "--model")
    if not model_arg:
        raise ValueError("--model is required for CUDA residency planning")
    model_path = Path(model_arg)
    config_path = model_path / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"model config is missing: {config_path}")
    try:
        model_config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid model config: {config_path}") from exc
    if not isinstance(model_config, dict):
        raise ValueError("model config must be a JSON object")
    model_config["model_path"] = str(model_path)
    model_config["config_sha256"] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"model weight index is missing: {index_path}")
    model_config["weight_index_sha256"] = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()
    revision = _arg_value(vllm_args, "--revision")
    if revision:
        model_config["revision"] = revision
    max_capture = int(
        _arg_value(vllm_args, "--max-cudagraph-capture-size")
        or os.getenv("VLLM_LATCHMOE_MAX_CAPTURE_SIZE", "32")
    )
    top_k = int(
        model_config.get(
            "num_experts_per_tok", model_config.get("num_selected_experts", 1)
        )
    )
    kv_reserve = int(os.getenv("VLLM_LATCHMOE_KV_RESERVE_BYTES", "0"))
    return build_residency_plan(
        plugin_args.offload_gib,
        model_config,
        max_capture_size=max_capture,
        top_k=top_k,
        device_total_bytes=_device_total_bytes(),
        kv_reserve_bytes=kv_reserve,
        profile_path=os.getenv("VLLM_LATCHMOE_PROFILE_GUIDED_PATH"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    plugin_args, vllm_args = parse_plugin_args(args)
    plan = build_plan_from_model_argument(plugin_args, vllm_args)
    os.environ["VLLM_LATCHMOE_MODE"] = "latchmoe"
    # Dynamic cache transitions are always outside CUDA Graph replay.  Eager is
    # retained only as an explicit correctness ablation.
    os.environ["VLLM_LATCHMOE_GRAPH_MODE"] = (
        "eager" if "--enforce-eager" in vllm_args else "piecewise"
    )
    os.environ["VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"] = serialize_residency_plan(plan)
    os.environ["VLLM_LATCHMOE_IDENTITY_LOCK_JSON"] = serialize_identity_lock(
        build_identity_lock(plan, vllm_args)
    )
    os.environ["VLLM_PLUGINS"] = "latchmoe_cuda"
    sys.argv = ["vllm", *vllm_args]
    from vllm.entrypoints.cli.main import main as vllm_main

    return int(vllm_main() or 0)
