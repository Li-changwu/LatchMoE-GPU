from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.library import Library
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

from .runtime import CudaLayerRuntime
from .runner_adapter import _apply_modular_moe_kernel, execute_exact_waves
from .routing import active_experts_from_topk
from .vllm_context import advance_moe_layer_index


STAGE_OP = "latchmoe::stage_experts"
FINISH_OP = "latchmoe::finish_experts"
GRAPH_SPLITTING_OPS = (STAGE_OP, FINISH_OP)

_graph_lib = Library("latchmoe", "FRAGMENT")


@dataclass(frozen=True)
class _GraphContext:
    runtime: CudaLayerRuntime
    experts_module: nn.Module


_contexts: dict[int, _GraphContext] = {}


def register_graph_runtime(runtime: CudaLayerRuntime, experts_module: nn.Module) -> int:
    context_id = runtime.layer_id
    previous = _contexts.get(context_id)
    if previous is not None and previous.runtime is not runtime:
        raise RuntimeError(f"duplicate LatchMoE graph layer id: {context_id}")
    _contexts[context_id] = _GraphContext(runtime, experts_module)
    return context_id


def _get_context(runtime_id: int) -> _GraphContext:
    try:
        return _contexts[runtime_id]
    except KeyError as exc:
        raise RuntimeError(f"unknown LatchMoE graph runtime id: {runtime_id}") from exc


def _stage_experts(
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    log2phy: torch.Tensor,
    topk_ids: torch.Tensor,
    runtime_id: int,
) -> None:
    context = _get_context(runtime_id)
    runtime = context.runtime
    advance_moe_layer_index(context.experts_module)
    if (
        slot_w13.data_ptr() != runtime.slot_w13.data_ptr()
        or slot_w2.data_ptr() != runtime.slot_w2.data_ptr()
        or log2phy.data_ptr() != runtime.log2phy.data_ptr()
    ):
        raise RuntimeError("graph staging tensors do not match the registered runtime")
    active = active_experts_from_topk(topk_ids)
    runtime.prepare_graph_compute(active)


def _stage_experts_fake(
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    log2phy: torch.Tensor,
    topk_ids: torch.Tensor,
    runtime_id: int,
) -> None:
    return None


def _fused_moe_compute(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    log2phy: torch.Tensor,
    runtime_id: int,
) -> torch.Tensor:
    context = _get_context(runtime_id)
    runtime = context.runtime
    if (
        slot_w13.data_ptr() != runtime.slot_w13.data_ptr()
        or slot_w2.data_ptr() != runtime.slot_w2.data_ptr()
        or log2phy.data_ptr() != runtime.log2phy.data_ptr()
    ):
        raise RuntimeError("graph compute tensors do not match the registered runtime")
    experts_module = context.experts_module
    overflow_active = runtime.graph_overflow_active
    if overflow_active is not None:

        def stage_kernel(w13, w2, pair_hidden, physical_ids, pair_weights):
            return _apply_modular_moe_kernel(
                experts_module,
                hidden_states=pair_hidden,
                topk_weights=pair_weights,
                topk_ids=physical_ids,
                w13=w13,
                w2=w2,
                global_num_experts=int(w13.shape[0]),
                expert_map=None,
            )

        return execute_exact_waves(
            runtime,
            hidden_states,
            topk_ids,
            topk_weights,
            stage_kernel_callback=stage_kernel,
            active_experts=overflow_active,
        )

    identity_slots = runtime.num_slots == runtime.num_experts
    return _apply_modular_moe_kernel(
        experts_module,
        hidden_states=hidden_states,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        w13=slot_w13,
        w2=slot_w2,
        global_num_experts=(
            runtime.num_slots if identity_slots else runtime.num_experts
        ),
        expert_map=None if identity_slots else log2phy,
    )


def _fused_moe_compute_fake(
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    log2phy: torch.Tensor,
    runtime_id: int,
) -> torch.Tensor:
    return torch.empty_like(hidden_states)


def _finish_experts(
    output: torch.Tensor,
    graph_token: torch.Tensor,
    runtime_id: int,
) -> None:
    del output
    runtime = _get_context(runtime_id).runtime
    runtime.finish_graph_compute()
    graph_token.add_(1)


def _finish_experts_fake(
    output: torch.Tensor,
    graph_token: torch.Tensor,
    runtime_id: int,
) -> None:
    return None


direct_register_custom_op(
    "stage_experts",
    _stage_experts,
    mutates_args=["slot_w13", "slot_w2", "log2phy"],
    fake_impl=_stage_experts_fake,
    target_lib=_graph_lib,
)
direct_register_custom_op(
    "fused_moe_compute",
    _fused_moe_compute,
    fake_impl=_fused_moe_compute_fake,
    target_lib=_graph_lib,
    tags=(torch.Tag.needs_fixed_stride_order,),
)
direct_register_custom_op(
    "finish_experts",
    _finish_experts,
    mutates_args=["graph_token"],
    fake_impl=_finish_experts_fake,
    target_lib=_graph_lib,
)


def graph_stage_experts(
    runtime: CudaLayerRuntime, runtime_id: int, topk_ids: torch.Tensor
) -> None:
    torch.ops.latchmoe.stage_experts(
        runtime.slot_w13,
        runtime.slot_w2,
        runtime.log2phy,
        topk_ids,
        runtime_id,
    )


def graph_finish_experts(
    runtime: CudaLayerRuntime, runtime_id: int, output: torch.Tensor
) -> None:
    torch.ops.latchmoe.finish_experts(output, runtime.graph_token, runtime_id)


def graph_fused_moe_compute(
    runtime: CudaLayerRuntime,
    runtime_id: int,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
) -> torch.Tensor:
    return torch.ops.latchmoe.fused_moe_compute(
        hidden_states,
        topk_weights,
        topk_ids,
        runtime.slot_w13,
        runtime.slot_w2,
        runtime.log2phy,
        runtime_id,
    )
