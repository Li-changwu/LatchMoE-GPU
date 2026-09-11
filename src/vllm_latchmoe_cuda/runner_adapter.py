from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .core.waves import plan_device_exact_waves, plan_main_cache_waves
from .capabilities import describe_capabilities, validate_capabilities
from .errors import NativeCombineError, StagingDuringCaptureError
from .moe_seam import CudaMoeSeam, FunctionalMoeSeam, NativeWavePayload
from .runtime import CudaLayerRuntime, WaveExecutionTrace
from .routing import active_experts_from_topk
from .split_ops import (
    eager_finish_compute,
    eager_needs_exact_waves,
    eager_prepare_compute,
)
from .vllm_context import advance_moe_layer_index


def _diagnostic_scatter_combine(*, waves, topk_weights, pair_offsets, restore_shape):
    """Oracle combine used only by legacy manifest tests.

    Production plan execution supplies the locked vLLM native combine hook.
    """
    output = torch.zeros(restore_shape, dtype=torch.float32, device=pair_offsets.device)
    for payload in waves:
        flat_weights = topk_weights.reshape(-1).float().index_select(
            0, payload.pair_offsets
        )
        weighted = payload.outputs.float() * flat_weights.unsqueeze(-1)
        output.scatter_add_(0, payload.token_indices.reshape(-1, 1).expand_as(weighted), weighted)
    return output.to(dtype=topk_weights.dtype)


@torch.compiler.disable
def execute_main_cache_waves(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    seam: CudaMoeSeam | None = None,
) -> torch.Tensor:
    """Serial hit-first execution through one persistent layer cache."""
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError("topk_ids and topk_weights must be matching rank-2 tensors")
    runtime.release_pending_for_transfer()
    active = active_experts_from_topk(topk_ids)
    ready = frozenset(
        expert
        for expert in active
        if runtime.main_cache.lease_for(expert) is not None
    )
    specs = plan_main_cache_waves(active, runtime.num_slots, hit_experts=ready)
    if seam is None:
        if getattr(runtime, "production_plan", False):
            raise NativeCombineError(
                "production LatchMoE requires the locked vLLM native combine seam"
            )
        seam = FunctionalMoeSeam(combine_fn=_diagnostic_scatter_combine)
    flat_ids = topk_ids.reshape(-1).long()
    pair_offsets = torch.arange(flat_ids.numel(), device=topk_ids.device, dtype=torch.long)
    payloads: list[NativeWavePayload] = []
    total_h2d_bytes = 0
    for spec in specs:
        prepared = runtime.prepare_main_cache_wave(spec)
        runtime.wait_and_publish(prepared)
        total_h2d_bytes += prepared.h2d_bytes
        runtime.begin_main_cache_compute(prepared)
        mask = torch.zeros_like(flat_ids, dtype=torch.bool)
        for expert in spec.experts:
            mask |= flat_ids == expert
        selected_offsets = pair_offsets[mask]
        token_indices = torch.div(
            selected_offsets, topk_ids.shape[1], rounding_mode="floor"
        )
        physical = runtime.log2phy.index_select(0, flat_ids[mask]).long()
        payload = seam.run_expert_mlp(
            hidden_states=hidden_states.index_select(0, token_indices),
            physical_ids=physical,
            slot_w13=runtime.slot_w13,
            slot_w2=runtime.slot_w2,
        )
        payloads.append(
            NativeWavePayload(
                outputs=payload.outputs,
                pair_offsets=selected_offsets,
                token_indices=token_indices,
            )
        )
        runtime.mark_compute_complete(prepared)
    result = seam.combine(
        waves=payloads,
        topk_weights=topk_weights,
        pair_offsets=pair_offsets,
        restore_shape=(hidden_states.shape[0], hidden_states.shape[1]),
    )
    runtime.last_wave_trace = WaveExecutionTrace(
        pair_count=int(pair_offsets.numel()),
        compute_order=tuple(spec.wave_id for spec in specs),
        issue_order=tuple(spec.wave_id for spec in specs),
        buffer_by_wave=tuple((spec.wave_id, 0) for spec in specs),
    )
    if runtime.event_writer is not None:
        runtime.event_writer.write(
            "main_cache_waves",
            layer_id=runtime.layer_id,
            pair_count=int(pair_offsets.numel()),
            wave_count=len(specs),
            stage_mode=[spec.wave_type for spec in specs],
            h2d_bytes=total_h2d_bytes,
            combine_count=1,
        )
    return result.to(dtype=hidden_states.dtype)


def capturable_slot_moe(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    selected_w13 = runtime.slot_w13[physical_topk_ids]
    gate_up = torch.einsum("tkoh,th->tko", selected_w13.float(), hidden_states.float())
    gate, up = gate_up.chunk(2, dim=-1)
    intermediate = F.silu(gate) * up
    selected_w2 = runtime.slot_w2[physical_topk_ids]
    expert_output = torch.einsum("tkoi,tki->tko", selected_w2.float(), intermediate)
    return (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)


def _capturable_weights_moe(
    w13: torch.Tensor,
    w2: torch.Tensor,
    hidden_states: torch.Tensor,
    physical_topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    selected_w13 = w13[physical_topk_ids]
    gate_up = torch.einsum("tkoh,th->tko", selected_w13.float(), hidden_states.float())
    gate, up = gate_up.chunk(2, dim=-1)
    selected_w2 = w2[physical_topk_ids]
    expert_output = torch.einsum(
        "tkoi,tki->tko", selected_w2.float(), F.silu(gate) * up
    )
    return (expert_output * topk_weights.unsqueeze(-1)).sum(dim=1)


def _resolve_modular_moe_kernel(experts_module: nn.Module):
    quant_method = experts_module.quant_method
    kernel = getattr(quant_method, "moe_kernel", None)
    if kernel is None:
        kernel = getattr(quant_method, "kernel", None)
    return kernel


def _apply_modular_moe_kernel(
    experts_module: nn.Module,
    *,
    hidden_states: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    w13: torch.Tensor,
    w2: torch.Tensor,
    global_num_experts: int,
    expert_map: torch.Tensor | None,
) -> torch.Tensor:
    kernel = _resolve_modular_moe_kernel(experts_module)
    if kernel is None:
        raise RuntimeError("LatchMoE requires the vLLM modular MoE kernel")
    result = kernel.apply(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        activation=experts_module.activation,
        global_num_experts=global_num_experts,
        expert_map=expert_map,
        apply_router_weight_on_input=experts_module.apply_router_weight_on_input,
        shared_experts_input=hidden_states,
    )
    if isinstance(result, tuple):
        raise TypeError("unexpected shared-expert result from Qwen3 routed experts")
    return result


def _validate_piecewise_slot_capacity(
    *,
    num_slots: int,
    num_experts: int,
    top_k: int,
    max_capture_size: int | None,
) -> int:
    if num_slots >= num_experts:
        return num_experts
    if not max_capture_size or top_k <= 0:
        raise RuntimeError(
            "finite-slot LatchMoE PIECEWISE requires finalized "
            "max_cudagraph_capture_size and FusedMoE.top_k"
        )
    required_slots = min(num_experts, int(max_capture_size) * top_k)
    if num_slots < required_slots:
        raise RuntimeError(
            "finite-slot LatchMoE PIECEWISE capture can overflow: "
            f"slots={num_slots}, required={required_slots}, "
            f"max_capture_size={max_capture_size}, top_k={top_k}"
        )
    return required_slots


@torch.compiler.disable
def execute_exact_waves(
    runtime: CudaLayerRuntime,
    hidden_states: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    *,
    transfer_aware: bool = True,
    kernel_callback=None,
    stage_kernel_callback=None,
    active_experts=None,
) -> torch.Tensor:
    # Compatibility name for legacy oracle callers. Production adapters call
    # execute_main_cache_waves directly; no temporary stage banks are created.
    if torch.cuda.is_current_stream_capturing():
        raise StagingDuringCaptureError(
            f"wave staging attempted during CUDA Graph capture: layer={runtime.layer_id}"
        )
    if runtime.stage_pool is None:
        return execute_main_cache_waves(runtime, hidden_states, topk_ids, topk_weights)
    if runtime.stage_pool is None:
        raise RuntimeError("exact waves require a CUDA stage pool")
    if kernel_callback is not None and stage_kernel_callback is not None:
        raise ValueError("only one exact-wave kernel callback may be provided")
    main_slots_overwritten = False
    if runtime.stage_pool.reuses_main_slots:
        runtime.main_slot_pool.acquire(
            runtime, runtime.stage_pool.transfer_engine.stream
        )
        main_slots_overwritten = True
    else:
        runtime.release_pending_for_transfer()
    plan = plan_device_exact_waves(
        topk_ids,
        topk_weights,
        capacity=runtime.stage_pool.num_slots,
        num_experts=runtime.num_experts,
        active_experts=active_experts,
    )
    bytes_per_expert = (
        runtime.host_w13[0].numel() * runtime.host_w13.element_size()
        + runtime.host_w2[0].numel() * runtime.host_w2.element_size()
    )
    h2d_bytes = {
        wave.wave_id: len(wave.experts) * bytes_per_expert for wave in plan.waves
    }
    preferred = (
        tuple(
            wave.wave_id
            for wave in sorted(
                plan.waves,
                key=lambda wave: (-h2d_bytes[wave.wave_id], wave.wave_id),
            )
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
    pair_outputs: list[torch.Tensor] = []
    scatter_indices: list[torch.Tensor] = []
    if kernel_callback is not None:
        runtime.acquire_main_slots_for_current_stream()

    def issue(wave_id: int) -> None:
        bank_id = free_banks.pop(0)
        issued[wave_id] = runtime.stage_pool.issue(
            runtime, wave_by_id[wave_id], bank_id
        )
        issue_log.append(wave_id)
        buffer_by_wave[wave_id] = bank_id

    try:
        for wave_id in plan.compute_order:
            if wave_id not in issued:
                issue(wave_id)
            for future in preferred:
                if not free_banks:
                    break
                if (
                    future != wave_id
                    and future not in issued
                    and future not in completed
                ):
                    issue(future)
            staged = issued.pop(wave_id)
            bank = runtime.stage_pool.wait_ready(staged)
            wave = wave_by_id[wave_id]
            pair_hidden = hidden_states.index_select(0, wave.token_indices)
            if stage_kernel_callback is not None:
                pair_output = stage_kernel_callback(
                    bank.w13,
                    bank.w2,
                    pair_hidden,
                    wave.physical_ids,
                    wave.pair_weights,
                )
            elif kernel_callback is None:
                pair_output = _capturable_weights_moe(
                    bank.w13,
                    bank.w2,
                    pair_hidden,
                    wave.physical_ids,
                    wave.pair_weights,
                )
            else:
                main_slots_overwritten = True
                wave_slots = int(bank.w13.shape[0])
                runtime.slot_w13.narrow(0, 0, wave_slots).copy_(bank.w13)
                runtime.slot_w2.narrow(0, 0, wave_slots).copy_(bank.w2)
                runtime.log2phy.copy_(wave.expert_map)
                pair_output = kernel_callback(
                    pair_hidden, wave.logical_ids, wave.pair_weights
                )
            pair_outputs.append(pair_output.float())
            scatter_indices.append(wave.token_indices)
            runtime.stage_pool.record_compute_done(bank.bank_id)
            if runtime.stage_pool.reuses_main_slots and bank.bank_id == 0:
                assert bank.compute_done is not None
                runtime.main_slot_pool.record_external_compute_done(bank.compute_done)
            completed.add(wave_id)
            free_banks.append(bank.bank_id)
            free_banks.sort()
    finally:
        if main_slots_overwritten:
            runtime.invalidate_main_slots()

    output.index_add_(0, torch.cat(scatter_indices), torch.cat(pair_outputs))

    runtime.last_wave_trace = WaveExecutionTrace(
        pair_count=plan.pair_count,
        compute_order=plan.compute_order,
        issue_order=tuple(issue_log),
        buffer_by_wave=tuple(sorted(buffer_by_wave.items())),
    )
    if runtime.event_writer is not None:
        runtime.event_writer.write(
            "exact_waves",
            layer_id=runtime.layer_id,
            pair_count=runtime.last_wave_trace.pair_count,
            wave_count=len(runtime.last_wave_trace.compute_order),
            compute_order=list(runtime.last_wave_trace.compute_order),
            issue_order=list(runtime.last_wave_trace.issue_order),
            pair_planner_mode="cuda_device",
            scatter_mode="layer_index_add",
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
    active = active_experts_from_topk(topk_ids)
    snapshot = runtime.stage_sync(active)
    handle = runtime.begin_compute(snapshot)
    try:
        physical_ids = runtime.log2phy[topk_ids.long()].long()
        return capturable_slot_moe(runtime, hidden_states, physical_ids, topk_weights)
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
    validate_capabilities(
        describe_capabilities(
            experts_module,
            graph_mode=("piecewise" if os.getenv("VLLM_LATCHMOE_GRAPH_MODE") == "piecewise" else "eager"),
        ),
        require_native_combine=False,
    )

    original_forward = experts_module.forward
    graph_mode = os.getenv("VLLM_LATCHMOE_GRAPH_MODE") == "piecewise"
    graph_runtime_id = None
    graph_stage = None
    graph_compute = None
    graph_finish = None
    if graph_mode:
        from vllm.config import get_cached_compilation_config

        from .graph_ops import (
            GRAPH_SPLITTING_OPS,
            graph_finish_experts,
            graph_fused_moe_compute,
            graph_stage_experts,
            register_graph_runtime,
        )

        compilation_config = get_cached_compilation_config()
        _validate_piecewise_slot_capacity(
            num_slots=runtime.num_slots,
            num_experts=runtime.num_experts,
            top_k=int(getattr(experts_module, "top_k", 0)),
            max_capture_size=compilation_config.max_cudagraph_capture_size,
        )
        if compilation_config.splitting_ops is None:
            compilation_config.splitting_ops = []
        for op in GRAPH_SPLITTING_OPS:
            if op not in compilation_config.splitting_ops:
                compilation_config.splitting_ops.append(op)
        graph_runtime_id = register_graph_runtime(runtime, experts_module)
        graph_stage = graph_stage_experts
        graph_compute = graph_fused_moe_compute
        graph_finish = graph_finish_experts

    def latchmoe_forward(hidden_states: torch.Tensor, router_logits: torch.Tensor):
        runtime.router_call_count += 1
        topk_weights, topk_ids = experts_module.router.select_experts(
            hidden_states=hidden_states,
            router_logits=router_logits,
        )
        if graph_mode:
            assert graph_runtime_id is not None
            assert (
                graph_stage is not None
                and graph_compute is not None
                and graph_finish is not None
            )
            graph_stage(runtime, graph_runtime_id, topk_ids)
            result = graph_compute(
                runtime,
                graph_runtime_id,
                hidden_states,
                topk_weights,
                topk_ids,
            )
            graph_finish(runtime, graph_runtime_id, result)
            return None, result
        advance_moe_layer_index(experts_module)
        if eager_needs_exact_waves(runtime, topk_ids):
            if _resolve_modular_moe_kernel(experts_module) is not None:

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

                result = execute_main_cache_waves(
                    runtime,
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    seam=getattr(experts_module, "_latchmoe_seam", None),
                )
            else:

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

                result = execute_main_cache_waves(
                    runtime,
                    hidden_states,
                    topk_ids,
                    topk_weights,
                    seam=getattr(experts_module, "_latchmoe_seam", None),
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
