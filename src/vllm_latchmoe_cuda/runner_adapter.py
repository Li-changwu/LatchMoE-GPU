from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch import nn

from .core.waves import plan_main_cache_waves
from .capabilities import describe_capabilities, validate_capabilities
from .errors import NativeCombineError, StagingDuringCaptureError
from .moe_seam import (
    CudaMoeSeam,
    FunctionalMoeSeam,
    NativeWavePayload,
    VllmModularMoeSeam,
)
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
    overlap: bool = True,
) -> torch.Tensor:
    """Serial hit-first execution through one persistent layer cache."""
    if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
        raise ValueError("topk_ids and topk_weights must be matching rank-2 tensors")
    runtime.ensure_healthy()
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
    prefetched: PreparedMainCacheWave | None = None
    for index, spec in enumerate(specs):
        try:
            if prefetched is not None:
                runtime.complete_pending_main_cache_computes()
                runtime.record_overlap(prefetched)
                prepared = prefetched
                prefetched = None
                runtime.wait_and_publish(prepared)
            else:
                prepared = runtime.prepare_main_cache_wave(spec)
                runtime.wait_and_publish(prepared)
            total_h2d_bytes += prepared.h2d_bytes
            runtime.begin_main_cache_compute(prepared)
        except BaseException as exc:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(runtime.device))
            runtime.poison(
                exc,
                wave_id=spec.wave_id,
                compute_done=end_event,
                transfer_tickets=tuple(runtime._pending_transfer_tickets.values()),
                active_experts=spec.experts,
            )
            raise
        try:
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
        except BaseException as exc:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(runtime.device))
            runtime.poison(
                exc,
                wave_id=spec.wave_id,
                compute_done=end_event,
                transfer_tickets=tuple(runtime._pending_transfer_tickets.values()),
                active_experts=spec.experts,
            )
            raise
        payloads.append(
            NativeWavePayload(
                outputs=payload.outputs,
                pair_offsets=selected_offsets,
                token_indices=token_indices,
            )
        )
        next_spec = specs[index + 1] if index + 1 < len(specs) else None
        if overlap and spec.overlap_candidate and next_spec is not None:
            try:
                prefetched = runtime.prepare_main_cache_wave(
                    next_spec,
                    protected_slots=frozenset(
                        lease.slot_id for lease in prepared.leases
                    ),
                    origin_event=runtime.compute_start_event(spec.wave_id),
                    overlap_candidate=spec.overlap_candidate,
                )
                runtime.mark_compute_complete(prepared, defer=True)
            except BaseException as exc:
                end_event = torch.cuda.Event(enable_timing=True)
                end_event.record(torch.cuda.current_stream(runtime.device))
                runtime.poison(
                    exc,
                    wave_id=spec.wave_id,
                    compute_done=end_event,
                    transfer_tickets=tuple(runtime._pending_transfer_tickets.values()),
                    active_experts=spec.experts,
                )
                raise
        else:
            runtime.mark_compute_complete(prepared)
    runtime.complete_pending_main_cache_computes()
    try:
        result = seam.combine(
            waves=payloads,
            topk_weights=topk_weights,
            pair_offsets=pair_offsets,
            restore_shape=(hidden_states.shape[0], hidden_states.shape[1]),
        )
    except BaseException as exc:
        runtime.poison(exc, active_experts=active)
        raise
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
    **_diagnostic_options,
) -> torch.Tensor:
    """Deprecated diagnostic alias for the serial main-cache executor."""
    if torch.cuda.is_current_stream_capturing():
        raise StagingDuringCaptureError(
            f"wave staging attempted during CUDA Graph capture: layer={runtime.layer_id}"
        )
    return execute_main_cache_waves(runtime, hidden_states, topk_ids, topk_weights)


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
    validate_capabilities(
        describe_capabilities(
            experts_module,
            graph_mode=("piecewise" if os.getenv("VLLM_LATCHMOE_GRAPH_MODE") == "piecewise" else "eager"),
            source_module=__import__(
                "vllm.model_executor.layers.fused_moe.fused_moe",
                fromlist=["fused_moe"],
            ),
        ),
        require_native_combine=False,
    )
    if getattr(runtime, "production_plan", False):
        seam = getattr(experts_module, "_latchmoe_seam", None)
        if seam is None:
            # vLLM 0.19.1 exposes the modular expert kernel and native
            # TopKWeightAndReduce primitive, but not a public layer-level seam.
            # Adapt those locked primitives to the wave ABI here.
            seam = VllmModularMoeSeam(
                experts_module=experts_module,
                runtime=runtime,
            )
            experts_module._latchmoe_seam = seam
        if not all(
            callable(getattr(seam, name, None))
            for name in ("run_expert_mlp", "combine")
        ):
            raise NativeCombineError(
                "qualified production runtime is missing the locked vLLM native combine seam"
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
        shared_experts = getattr(experts_module, "_shared_experts", None)
        shared_output = (
            shared_experts(hidden_states) if shared_experts is not None else None
        )
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
            return shared_output, result
        advance_moe_layer_index(experts_module)
        if eager_needs_exact_waves(runtime, topk_ids):
            result = execute_main_cache_waves(
                runtime,
                hidden_states,
                topk_ids,
                topk_weights,
                seam=getattr(experts_module, "_latchmoe_seam", None),
                overlap=os.getenv("VLLM_LATCHMOE_OVERLAP", "1") != "0",
            )
            return shared_output, result
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
        return shared_output, result

    experts_module._latchmoe_original_forward = original_forward
    experts_module.forward = latchmoe_forward
    experts_module._latchmoe_forward_installed = True
