from __future__ import annotations

import torch

from .runtime import CudaLayerRuntime, LayerMappingSnapshot
from .routing import active_experts_from_topk


@torch.compiler.disable
def eager_stage_and_map(
    runtime: CudaLayerRuntime, topk_ids: torch.Tensor
) -> LayerMappingSnapshot:
    active = active_experts_from_topk(topk_ids)
    return runtime.stage_async(active)


@torch.compiler.disable
def eager_prepare_compute(runtime: CudaLayerRuntime, topk_ids: torch.Tensor) -> None:
    active = active_experts_from_topk(topk_ids)
    if runtime.event_writer is not None and not runtime.direct_slots_profiled:
        runtime.event_writer.write(
            "direct_slots",
            layer_id=runtime.layer_id,
            active_experts=len(active),
            slot_capacity=runtime.num_slots,
        )
        runtime.direct_slots_profiled = True
    runtime.prepare_compute_async(active)


@torch.compiler.disable
def eager_finish_compute(runtime: CudaLayerRuntime) -> None:
    runtime.finish_compute_async()


@torch.compiler.disable
def eager_needs_exact_waves(runtime: CudaLayerRuntime, topk_ids: torch.Tensor) -> bool:
    return len(active_experts_from_topk(topk_ids)) > runtime.num_slots
