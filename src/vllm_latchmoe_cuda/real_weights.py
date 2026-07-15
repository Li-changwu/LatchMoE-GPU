from __future__ import annotations

import gc
import json
from dataclasses import replace
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .manifest import LayerLayout, OffloadManifest
from .offloader import CudaSEWOffloader
from .runner_adapter import (
    _capturable_weights_moe,
    eager_slot_moe,
    execute_exact_waves,
)


def layer_checkpoint_keys(layer_id: int, *, num_experts: int) -> tuple[str, ...]:
    prefix = f"model.layers.{layer_id}.mlp.experts"
    return tuple(
        f"{prefix}.{expert_id}.{projection}.weight"
        for expert_id in range(num_experts)
        for projection in ("gate_proj", "up_proj", "down_proj")
    )


class _RealExperts(nn.Module):
    def __init__(self, layer: LayerLayout, device: torch.device):
        super().__init__()
        w13 = layer.tensor("w13_weight")
        w2 = layer.tensor("w2_weight")
        self.w13_weight = nn.Parameter(
            torch.empty(w13.shape, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty(w2.shape, dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


class _RealMlp(nn.Module):
    def __init__(self, layer: LayerLayout, device: torch.device):
        super().__init__()
        self.experts = _RealExperts(layer, device)


class _RealDecoder(nn.Module):
    def __init__(self, layer: LayerLayout, device: torch.device):
        super().__init__()
        self.mlp = _RealMlp(layer, device)


def _single_layer_manifest(manifest: OffloadManifest, layer_id: int) -> OffloadManifest:
    source = manifest.layer(layer_id)
    base = min(tensor.offset_elements for tensor in source.tensors)
    layer = LayerLayout(
        layer_id=layer_id,
        tensors=tuple(
            replace(tensor, offset_elements=tensor.offset_elements - base)
            for tensor in source.tensors
        ),
    )
    return replace(manifest, layers=(layer,))


def _load_layer_into_host_store(
    manifest: OffloadManifest,
    layer_id: int,
    offloader: CudaSEWOffloader,
) -> None:
    model_path = Path(manifest.model.path)
    index = json.loads(
        (model_path / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]
    host_w13 = offloader.host_store.tensor_view(layer_id, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(layer_id, "w2_weight")
    keys_by_shard: dict[str, list[str]] = {}
    for key in layer_checkpoint_keys(layer_id, num_experts=manifest.model.num_experts):
        keys_by_shard.setdefault(index[key], []).append(key)

    prefix = f"model.layers.{layer_id}.mlp.experts."
    intermediate = host_w2.shape[-1]
    for shard, keys in keys_by_shard.items():
        with safe_open(model_path / shard, framework="pt", device="cpu") as handle:
            for key in keys:
                remainder = key.removeprefix(prefix)
                expert_text, projection, _ = remainder.split(".")
                expert_id = int(expert_text)
                weight = handle.get_tensor(key)
                if projection == "gate_proj":
                    host_w13[expert_id, :intermediate].copy_(weight)
                elif projection == "up_proj":
                    host_w13[expert_id, intermediate:].copy_(weight)
                elif projection == "down_proj":
                    host_w2[expert_id].copy_(weight)
                else:
                    raise AssertionError(f"unexpected projection in {key}")


def _comparison(actual: torch.Tensor, expected: torch.Tensor) -> dict[str, object]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    delta = (actual_fp32 - expected_fp32).abs()
    rtol = 2e-2
    atol = 2e-2
    return {
        "close": bool(torch.allclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol)),
        "max_abs": float(delta.max().item()),
        "mean_abs": float(delta.mean().item()),
        "rtol": rtol,
        "atol": atol,
    }


def compare_qwen_layer(
    *, manifest_path: str | Path, layer_id: int, artifact_path: str | Path
) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    manifest = OffloadManifest.load(manifest_path)
    manifest.validate_model_files()
    single = _single_layer_manifest(manifest, layer_id)
    device = torch.device("cuda")
    module = _RealDecoder(single.layers[0], device)
    offloader = CudaSEWOffloader(single, first_layer_id=layer_id)
    offloader.wrap_modules(iter((module,)))
    _load_layer_into_host_store(manifest, layer_id, offloader)
    host_w13 = offloader.host_store.tensor_view(layer_id, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(layer_id, "w2_weight")
    full_w13 = host_w13.to(device)
    full_w2 = host_w2.to(device)
    offloader.post_init()
    runtime = offloader.runtimes[layer_id]
    torch.manual_seed(1000 + layer_id)

    eager_ids = torch.arange(32, device=device).view(4, 8)
    eager_weights = torch.rand((4, 8), device=device)
    eager_weights /= eager_weights.sum(dim=-1, keepdim=True)
    eager_hidden = torch.randn((4, 2048), dtype=torch.bfloat16, device=device)
    eager_expected = _capturable_weights_moe(
        full_w13, full_w2, eager_hidden, eager_ids, eager_weights
    )
    eager_actual = eager_slot_moe(runtime, eager_hidden, eager_ids, eager_weights)

    full_union_ids = torch.arange(128, device=device).view(16, 8)
    full_union_weights = torch.rand((16, 8), device=device)
    full_union_weights /= full_union_weights.sum(dim=-1, keepdim=True)
    full_union_hidden = torch.randn((16, 2048), dtype=torch.bfloat16, device=device)
    full_union_expected = _capturable_weights_moe(
        full_w13,
        full_w2,
        full_union_hidden,
        full_union_ids,
        full_union_weights,
    )
    if runtime.stage_pool is None:
        full_union_actual = eager_slot_moe(
            runtime, full_union_hidden, full_union_ids, full_union_weights
        )
        full_union_mode = "identity_slots"
    else:
        full_union_actual = execute_exact_waves(
            runtime, full_union_hidden, full_union_ids, full_union_weights
        )
        full_union_mode = "exact_waves"
    torch.cuda.synchronize()

    eager = _comparison(eager_actual, eager_expected)
    full_union = _comparison(full_union_actual, full_union_expected)
    result: dict[str, object] = {
        "layer_id": layer_id,
        "manifest_sha256": manifest.manifest_sha256,
        "eager_close": eager["close"],
        "eager_max_abs": eager["max_abs"],
        "eager_mean_abs": eager["mean_abs"],
        "full_union_close": full_union["close"],
        "full_union_max_abs": full_union["max_abs"],
        "full_union_mean_abs": full_union["mean_abs"],
        "full_union_mode": full_union_mode,
        "rtol": eager["rtol"],
        "atol": eager["atol"],
    }
    if runtime.last_wave_trace is not None:
        result.update(
            wave_count=len(runtime.last_wave_trace.compute_order),
            wave_pair_count=runtime.last_wave_trace.pair_count,
        )
    artifact_path = Path(artifact_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    del full_w13, full_w2, module, offloader, runtime
    gc.collect()
    torch.cuda.empty_cache()
    return result
