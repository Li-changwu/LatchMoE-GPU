from __future__ import annotations

import torch

from .runtime import CudaLayerRuntime, LayerMappingSnapshot


@torch.compiler.disable
def eager_stage_and_map(
    runtime: CudaLayerRuntime, topk_ids: torch.Tensor
) -> LayerMappingSnapshot:
    active = tuple(int(value) for value in torch.unique(topk_ids).cpu().tolist())
    return runtime.stage_async(active)


@torch.compiler.disable
def eager_prepare_compute(runtime: CudaLayerRuntime, topk_ids: torch.Tensor) -> None:
    active = tuple(int(value) for value in torch.unique(topk_ids).cpu().tolist())
    runtime.prepare_compute_async(active)


@torch.compiler.disable
def eager_finish_compute(runtime: CudaLayerRuntime) -> None:
    runtime.finish_compute_async()


@torch.compiler.disable
def eager_needs_exact_waves(runtime: CudaLayerRuntime, topk_ids: torch.Tensor) -> bool:
    return int(torch.unique(topk_ids).numel()) > runtime.num_slots
