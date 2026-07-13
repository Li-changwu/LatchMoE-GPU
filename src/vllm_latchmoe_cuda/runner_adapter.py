from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .core.waves import (
    WaveDescriptor,
    plan_exact_waves,
    plan_transfer_issue_order,
    validate_pair_coverage,
)
from .runtime import CudaLayerRuntime, WaveExecutionTrace
from .split_ops import (
    eager_finish_compute,
    eager_needs_exact_waves,
    eager_prepare_compute,
)


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


def _capturable_weights_moe(
    w13: torch.Tensor,
    w2: torch.Tensor,
    hidden_states: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    selected_w13 = w13[physical_topk_ids]
    gate_up = torch.einsum(
        "tkoh,th->tko", selected_w13.float(), hidden_states.float()
    )
    gate, up = gate_up.chunk(2, dim=-1)
    selected_w2 = w2[physical_topk_ids]
    expert_output = torch.einsum(
        "tkoi,tki->tko", selected_w2.float(), F.silu(gate) * up
    )
    return (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)


def execute_exact_waves(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    transfer_aware: bool = True,
    kernel_callback=None,
) -> torch.Tensor:
    runtime.release_pending_for_transfer()
    plan = plan_exact_waves(topk_ids, topk_weights, capacity=runtime.num_slots)
    validate_pair_coverage(plan, expected_pairs=topk_ids.numel())
    bytes_per_expert = (
        runtime.host_w13[0].numel() * runtime.host_w13.element_size()
        + runtime.host_w2[0].numel() * runtime.host_w2.element_size()
    )
    h2d_bytes = {
        wave.wave_id: len(wave.experts) * bytes_per_expert for wave in plan.waves
    }
    preferred = (
        plan_transfer_issue_order(
            plan.waves,
            ready_experts=frozenset(),
            h2d_bytes_by_wave=h2d_bytes,
        )
        if transfer_aware
        else plan.compute_order
    )
    wave_by_id = {wave.wave_id: wave for wave in plan.waves}
    issued: dict[int, object] = {}
    completed: set[int] = set()
    free_banks = [0, 1]
    issue_log: list[int] = []
    buffer_by_wave: dict[int, int] = {}
    output = torch.zeros(
        (hidden_states.shape[0], hidden_states.shape[1]),
        dtype=torch.float32,
        device=hidden_states.device,
    )

    def issue(wave_id: int) -> None:
        bank_id = free_banks.pop(0)
        issued[wave_id] = runtime.stage_pool.issue(
            runtime, wave_by_id[wave_id], bank_id
        )
        issue_log.append(wave_id)
        buffer_by_wave[wave_id] = bank_id

    for wave_id in plan.compute_order:
        if wave_id not in issued:
            issue(wave_id)
        for future in preferred:
            if not free_banks:
                break
            if future != wave_id and future not in issued and future not in completed:
                issue(future)
        staged = issued.pop(wave_id)
        bank = runtime.stage_pool.wait_ready(staged)
        wave = wave_by_id[wave_id]
        token_indices = torch.tensor(
            [pair.token_index for pair in wave.pairs],
            dtype=torch.long,
            device=hidden_states.device,
        )
        logical_ids = torch.tensor(
            [[pair.expert_id] for pair in wave.pairs],
            dtype=torch.long,
            device=hidden_states.device,
        )
        physical_by_expert = {
            expert: position for position, expert in enumerate(wave.experts)
        }
        physical_ids = torch.tensor(
            [[physical_by_expert[pair.expert_id]] for pair in wave.pairs],
            dtype=torch.long,
            device=hidden_states.device,
        )
        pair_weights = torch.tensor(
            [[pair.weight] for pair in wave.pairs],
            dtype=topk_weights.dtype,
            device=hidden_states.device,
        )
        pair_hidden = hidden_states.index_select(0, token_indices)
        if kernel_callback is None:
            pair_output = _capturable_weights_moe(
                bank.w13, bank.w2, pair_hidden, physical_ids, pair_weights
            )
        else:
            runtime.slot_w13.copy_(bank.w13)
            runtime.slot_w2.copy_(bank.w2)
            wave_map = torch.full_like(runtime.log2phy, -1)
            for expert, position in physical_by_expert.items():
                wave_map[expert] = position
            runtime.log2phy.copy_(wave_map)
            pair_output = kernel_callback(pair_hidden, logical_ids, pair_weights)
        output.index_add_(0, token_indices, pair_output.float())
        runtime.stage_pool.record_compute_done(bank.bank_id)
        completed.add(wave_id)
        free_banks.append(bank.bank_id)
        free_banks.sort()

    runtime.last_wave_trace = WaveExecutionTrace(
        pair_count=topk_ids.numel(),
        compute_order=plan.compute_order,
        issue_order=tuple(issue_log),
        buffer_by_wave=tuple(sorted(buffer_by_wave.items())),
    )
    return output.to(dtype=hidden_states.dtype)


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
        if eager_needs_exact_waves(runtime, topk_ids):

            def original_kernel(pair_hidden, logical_ids, pair_weights):
                pair_result = experts_module.quant_method.apply(
                    layer=experts_module,
                    x=pair_hidden,
                    topk_weights=pair_weights,
                    topk_ids=logical_ids,
                    shared_experts_input=pair_hidden,
                )
                if isinstance(pair_result, tuple):
                    raise TypeError(
                        "unexpected shared-expert result from Qwen3 routed experts"
                    )
                return pair_result

            result = execute_exact_waves(
                runtime,
                hidden_states,
                topk_ids,
                topk_weights,
                kernel_callback=original_kernel,
            )
            return None, result
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
