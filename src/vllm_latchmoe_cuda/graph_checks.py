from __future__ import annotations

from unittest import mock
from typing import Any, Mapping

import torch
from torch import nn

from .errors import StagingDuringCaptureError
from .manifest import (
    LayerLayout,
    ModelIdentity,
    OffloadManifest,
    TensorLayout,
    VllmIdentity,
)
from .offloader import CudaSEWOffloader
from .runner_adapter import capturable_slot_moe
from .split_ops import eager_stage_and_map


class GraphCheckError(RuntimeError):
    pass


def validate_graph_report(report: Mapping[str, Any]) -> dict[str, Any]:
    mode = report.get("mode")
    if mode not in {"eager", "piecewise"}:
        raise GraphCheckError(f"unknown graph check mode: {mode!r}")
    if not report.get("addresses_stable"):
        raise GraphCheckError("persistent tensor address changed")
    if not report.get("staging_during_capture_rejected"):
        raise GraphCheckError("dynamic staging was not rejected during capture")
    if int(report.get("graph_break_count", 0)) < 1:
        raise GraphCheckError("eager split produced no graph break")
    if mode == "piecewise" and int(report.get("captured_segment_count", 0)) < 1:
        raise GraphCheckError("piecewise check has no captured segment")
    growth = int(report.get("allocated_growth_bytes", 0))
    if growth < 0:
        raise GraphCheckError("allocated memory growth cannot be negative")
    growth_limit = int(report.get("allocated_growth_limit_bytes", 0))
    if growth_limit <= 0:
        raise GraphCheckError("allocator growth limit must be positive")
    if growth > growth_limit:
        raise GraphCheckError(
            f"allocator growth exceeds limit: growth={growth}, limit={growth_limit}"
        )
    reserved_growth = int(report.get("reserved_growth_bytes", 0))
    reserved_limit = int(report.get("reserved_growth_limit_bytes", 0))
    if reserved_growth < 0 or reserved_limit <= 0:
        raise GraphCheckError("reserved memory growth bounds are invalid")
    if reserved_growth > reserved_limit:
        raise GraphCheckError(
            f"reserved allocator growth exceeds limit: growth={reserved_growth}, "
            f"limit={reserved_limit}"
        )
    return dict(report)


class _SyntheticExperts(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty((4, 4, 2), dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty((4, 2, 2), dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )


class _SyntheticMlp(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.experts = _SyntheticExperts(device)


class _SyntheticDecoder(nn.Module):
    def __init__(self, device: torch.device):
        super().__init__()
        self.mlp = _SyntheticMlp(device)


def _synthetic_manifest() -> OffloadManifest:
    return OffloadManifest(
        schema_version=1,
        manifest_sha256="synthetic-graph-probe",
        model=ModelIdentity(
            path="synthetic",
            revision="synthetic",
            config_sha256="0" * 64,
            weight_index_sha256="0" * 64,
            num_experts=4,
        ),
        vllm=VllmIdentity(
            version="0.19.1",
            tag_commit="b1388b1fbf5aaef47937fabe98931211684666a6",
        ),
        dtype="bfloat16",
        tensor_parallel_size=1,
        num_slots=2,
        layers=(
            LayerLayout(
                layer_id=0,
                tensors=(
                    TensorLayout(
                        name="w13_weight",
                        parameter_name="mlp.experts.w13_weight",
                        shape=(4, 4, 2),
                        stride=(8, 2, 1),
                        dtype="bfloat16",
                        offset_elements=0,
                        numel=32,
                        nbytes=64,
                    ),
                    TensorLayout(
                        name="w2_weight",
                        parameter_name="mlp.experts.w2_weight",
                        shape=(4, 2, 2),
                        stride=(4, 2, 1),
                        dtype="bfloat16",
                        offset_elements=32,
                        numel=16,
                        nbytes=32,
                    ),
                ),
            ),
        ),
    )


def run_synthetic_graph_probe(mode: str) -> dict[str, Any]:
    if mode not in {"eager", "piecewise"}:
        raise ValueError(f"unknown graph probe mode: {mode!r}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device = torch.device("cuda")
    module = _SyntheticDecoder(device)
    offloader = CudaSEWOffloader(_synthetic_manifest())
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(71)
    host_w13 = offloader.host_store.tensor_view(0, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(0, "w2_weight")
    host_w13.copy_(torch.randn_like(host_w13))
    host_w2.copy_(torch.randn_like(host_w2))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device=device)
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32, device=device
    )
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device=device)

    runtime.stage_sync((0, 1))
    physical_ids = runtime.log2phy[topk_ids].long()
    expected = capturable_slot_moe(runtime, hidden, physical_ids, topk_weights)
    torch.cuda.synchronize()
    pointers_before = runtime.data_ptrs()

    def split_function(x, ids):
        before = x + 1
        eager_stage_and_map(runtime, ids)
        return before * 2

    explanation = torch._dynamo.explain(split_function)(
        torch.ones((1,), device=device), topk_ids
    )

    staging_rejected = False
    with mock.patch.object(
        torch.cuda, "is_current_stream_capturing", return_value=True
    ):
        try:
            runtime.stage_async((0, 1))
        except StagingDuringCaptureError:
            staging_rejected = True

    for step in range(10):
        runtime.stage_sync((step % 4, (step + 1) % 4))
    torch.cuda.synchronize()
    allocated_before = torch.cuda.memory_allocated(device)
    reserved_before = torch.cuda.memory_reserved(device)
    for step in range(100):
        runtime.stage_sync((step % 4, (step + 1) % 4))
    torch.cuda.synchronize()
    allocated_after = torch.cuda.memory_allocated(device)
    reserved_after = torch.cuda.memory_reserved(device)

    runtime.stage_sync((0, 1))
    physical_ids = runtime.log2phy[topk_ids].long()
    expected = capturable_slot_moe(runtime, hidden, physical_ids, topk_weights)
    captured_segment_count = 0
    actual = expected
    if mode == "piecewise":
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = capturable_slot_moe(runtime, hidden, physical_ids, topk_weights)
        graph.replay()
        torch.cuda.synchronize()
        captured_segment_count = 1

    pointers_after = runtime.data_ptrs()
    report = {
        "schema_version": 1,
        "scope": "synthetic-runtime",
        "mode": mode,
        "device": torch.cuda.get_device_name(device),
        "addresses_stable": pointers_before == pointers_after,
        "pointers_before": pointers_before,
        "pointers_after": pointers_after,
        "staging_during_capture_rejected": staging_rejected,
        "capture_guard_method": "forced-is_current_stream_capturing",
        "graph_count": int(explanation.graph_count),
        "graph_break_count": int(explanation.graph_break_count),
        "captured_segment_count": captured_segment_count,
        "allocated_before_bytes": allocated_before,
        "allocated_after_bytes": allocated_after,
        "allocated_delta_bytes": allocated_after - allocated_before,
        "allocated_growth_bytes": max(0, allocated_after - allocated_before),
        "allocated_growth_limit_bytes": 1024 * 1024,
        "reserved_before_bytes": reserved_before,
        "reserved_after_bytes": reserved_after,
        "reserved_delta_bytes": reserved_after - reserved_before,
        "reserved_growth_bytes": max(0, reserved_after - reserved_before),
        "reserved_growth_limit_bytes": 1024 * 1024,
        "output_close": bool(torch.allclose(actual, expected, rtol=2e-2, atol=2e-2)),
    }
    validate_graph_report(report)
    if not report["output_close"]:
        raise GraphCheckError("captured output differs from eager output")
    return report
