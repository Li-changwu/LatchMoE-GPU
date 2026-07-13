from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .runtime import CudaLayerRuntime
from .split_ops import eager_finish_compute, eager_prepare_compute


def capturable_slot_moe(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    selected_w13 = runtime.slot_w13[physical_topk_ids]
    gate_up = torch.einsum(
        "tkoh,th->tko", selected_w13.float(), hidden_states.float()
    )
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    selected_w2 = runtime.slot_w2[physical_topk_ids]
    expert_output = torch.einsum(
        "tkoi,tki->tko", selected_w2.float(), intermediate
    )
    return (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)


def eager_slot_moe(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have identical shapes")
    if hidden_states.shape[0] != topk_ids.shape[0]:
        raise ValueError("routing row count must equal hidden-state row count")
    active = tuple(int(value) for value in torch.unique(topk_ids).cpu().tolist())
    snapshot = runtime.stage_sync(active)
    handle = runtime.begin_compute(snapshot)
    try:
        physical_ids = runtime.log2phy[topk_ids.long()].long()
        return capturable_slot_moe(
            runtime, hidden_states, physical_ids, topk_weights
        )
    finally:
        runtime.end_compute(handle)


def install_vllm_forward_adapter(
    experts_module: nn.Module, runtime: CudaLayerRuntime
) -> None:
    if getattr(experts_module, "_latchmoe_forward_installed", False):
        return
    router = getattr(experts_module, "router", None)
    quant_method = getattr(experts_module, "quant_method", None)
    if router is None or not hasattr(router, "select_experts"):
        raise TypeError("FusedMoE router.select_experts is required")
    if quant_method is None or not hasattr(quant_method, "apply"):
        raise TypeError("FusedMoE quant_method.apply is required")
    if bool(getattr(quant_method, "is_monolithic", False)):
        raise TypeError("monolithic FusedMoE kernels are not supported")
    if getattr(experts_module, "_shared_experts", None) is not None:
        raise TypeError("shared experts are not supported by the Qwen3 target adapter")

    original_forward = experts_module.forward

    def latchmoe_forward(hidden_states: torch.Tensor, router_logits: torch.Tensor):
        topk_weights, topk_ids = experts_module.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        eager_prepare_compute(runtime, topk_ids)
        try:
            result = experts_module.quant_method.apply(
                layer=experts_module,
                x=hidden_states,
                topk_weights=topk_weights,
                topk_ids=topk_ids,
                shared_experts_input=hidden_states,
            )
        finally:
            eager_finish_compute(runtime)
        if isinstance(result, tuple):
            raise TypeError("unexpected shared-expert result from Qwen3 routed experts")
        return None, result

    experts_module._latchmoe_original_forward = original_forward
    experts_module.forward = latchmoe_forward
    experts_module._latchmoe_forward_installed = True
