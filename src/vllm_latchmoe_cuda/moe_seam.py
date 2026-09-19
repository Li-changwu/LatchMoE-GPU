"""Narrow dispatch/compute/combine seam used by main-cache execution."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, Sequence

import torch

from .errors import NativeCombineError


@dataclass(frozen=True)
class NativeWavePayload:
    """Expert outputs before layer-level top-k reduction."""

    outputs: torch.Tensor
    pair_offsets: torch.Tensor
    token_indices: torch.Tensor


class CudaMoeSeam(Protocol):
    def run_expert_mlp(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        logical_ids: torch.Tensor,
        physical_ids: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
    ) -> NativeWavePayload: ...

    def combine(
        self,
        *,
        waves: Sequence[NativeWavePayload],
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        restore_shape: tuple[int, ...],
    ) -> torch.Tensor: ...


class FunctionalMoeSeam:
    """Small seam for tests and the qualified routed-only fallback.

    A caller may inject the locked vLLM combine primitive.  Without one we
    fail closed instead of silently presenting a scatter implementation as a
    native combine.
    """

    def __init__(
        self,
        *,
        combine_fn: Callable[..., torch.Tensor] | None = None,
        activation: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ):
        self.combine_fn = combine_fn
        self.activation = activation or torch.nn.functional.silu

    def run_expert_mlp(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        logical_ids: torch.Tensor,
        physical_ids: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
    ) -> NativeWavePayload:
        del topk_weights
        ids = physical_ids.reshape(-1).long()
        weights = slot_w13.index_select(0, ids).float()
        token_indices = torch.div(
            pair_offsets,
            topk_ids.shape[1],
            rounding_mode="floor",
        )
        hidden = hidden_states.index_select(0, token_indices).float()
        gate_up = torch.einsum("pih,ph->pi", weights, hidden)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.activation(gate) * up
        down = slot_w2.index_select(0, ids).float()
        outputs = torch.einsum("poi,pi->po", down, intermediate)
        return NativeWavePayload(
            outputs=outputs,
            pair_offsets=torch.empty(0, dtype=torch.long, device=outputs.device),
            token_indices=token_indices,
        )

    def combine(
        self,
        *,
        waves: Sequence[NativeWavePayload],
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        restore_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if self.combine_fn is None:
            raise NativeCombineError(
                "vLLM native combine primitive was not supplied by the locked seam"
            )
        return self.combine_fn(
            waves=waves,
            topk_weights=topk_weights,
            pair_offsets=pair_offsets,
            restore_shape=restore_shape,
        )


class VllmModularMoeSeam:
    """Adapt vLLM's locked modular kernel to one Main Cache wave.

    The wheel does not expose a public layer-level combine callback.  Its
    modular Triton kernel exposes the primitives and workspaces we need.  Each
    wave retains the original ``[num_tokens, topk]`` routing shape and masks
    non-resident experts through an expert map.  We copy the weighted outputs
    for that wave out of Triton's pair workspace, then run one native reduction
    after all waves.  This preserves the exact-UVA kernel configuration and
    arithmetic order.
    """

    def __init__(self, *, experts_module, runtime):
        self.experts_module = experts_module
        self.runtime = runtime
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceContiguous,
        )

        self._reducer = TopKWeightAndReduceContiguous()

    def run_expert_mlp(
        self,
        *,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        logical_ids: torch.Tensor,
        physical_ids: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
    ) -> NativeWavePayload:
        quant_method = getattr(self.experts_module, "quant_method", None)
        kernel = getattr(quant_method, "moe_kernel", None)
        if kernel is None:
            kernel = getattr(quant_method, "kernel", None)
        kernel_impl = getattr(kernel, "impl", kernel)
        prepare = getattr(kernel_impl, "_prepare", None)
        allocate = getattr(kernel_impl, "_allocate_buffers", None)
        fused_experts = getattr(kernel_impl, "fused_experts", None)
        expert_apply = getattr(fused_experts, "apply", None)
        if not all(callable(value) for value in (prepare, allocate, expert_apply)):
            raise NativeCombineError(
                "vLLM modular Triton kernel is missing the locked wave ABI"
            )
        if type(fused_experts).__name__ != "TritonExperts":
            raise NativeCombineError(
                "exact multi-wave execution requires vLLM TritonExperts, got "
                f"{type(fused_experts).__name__}"
            )
        if bool(getattr(self.experts_module, "apply_router_weight_on_input", False)):
            raise NativeCombineError(
                "exact multi-wave execution does not support router weights on input"
            )
        if topk_ids.shape != topk_weights.shape or topk_ids.ndim != 2:
            raise NativeCombineError("native routing tensors must be matching rank-2")
        if hidden_states.shape[0] != topk_ids.shape[0]:
            raise NativeCombineError("native routing rows must match hidden-state rows")
        if pair_offsets.numel() != logical_ids.numel():
            raise NativeCombineError("wave pair offsets and logical IDs differ in size")
        routed_ids = topk_ids.reshape(-1).index_select(0, pair_offsets.long())
        if not torch.equal(routed_ids.long(), logical_ids.reshape(-1).long()):
            raise NativeCombineError("wave pair offsets do not match native routing IDs")

        wave_map = torch.full(
            (self.runtime.num_experts,),
            -1,
            dtype=self.runtime.log2phy.dtype,
            device=hidden_states.device,
        )
        wave_logical_ids = torch.unique(logical_ids.reshape(-1).long())
        wave_physical_ids = self.runtime.log2phy.index_select(
            0, wave_logical_ids
        )
        if bool(torch.any(wave_physical_ids < 0).item()):
            raise NativeCombineError("wave contains an expert without a resident slot")
        if not torch.equal(
            self.runtime.log2phy.index_select(0, logical_ids.reshape(-1).long()).long(),
            physical_ids.reshape(-1).long(),
        ):
            raise NativeCombineError("wave physical IDs do not match the runtime map")
        wave_map.index_copy_(0, wave_logical_ids, wave_physical_ids)

        global_num_experts = int(self.runtime.num_experts)
        a1q, a1q_scale, expert_tokens_meta, prepared_ids, prepared_weights = prepare(
            hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            global_num_experts=global_num_experts,
            expert_map=wave_map,
            apply_router_weight_on_input=False,
        )
        # No-DP/EP preparation must preserve the native routing layout.  A
        # gathered or permuted layout would invalidate pair-offset recovery.
        if (
            prepared_ids.shape != topk_ids.shape
            or prepared_weights.shape != topk_weights.shape
            or not torch.equal(prepared_ids, topk_ids)
            or not torch.equal(prepared_weights, topk_weights)
        ):
            raise NativeCombineError(
                "vLLM preparation changed the native routing layout"
            )
        local_num_experts = int(slot_w13.shape[0])
        problem_size = fused_experts.moe_problem_size(
            a1q, slot_w13, slot_w2, prepared_ids
        )
        _, num_tokens, intermediate_size, hidden_size, topk = problem_size
        workspace13, workspace2, output = allocate(
            hidden_states.dtype,
            hidden_states.device,
            num_tokens,
            num_tokens,
            intermediate_size,
            hidden_size,
            topk,
            global_num_experts,
            local_num_experts,
            expert_tokens_meta,
            self.experts_module.activation,
        )
        required_pair_values = num_tokens * topk * hidden_size
        if (
            workspace2.numel() < required_pair_values
            or output.shape != hidden_states.shape
        ):
            raise NativeCombineError("vLLM Triton workspace has an incompatible shape")
        expert_apply(
            output=output,
            hidden_states=a1q,
            w1=slot_w13,
            w2=slot_w2,
            topk_weights=prepared_weights,
            topk_ids=prepared_ids,
            activation=self.experts_module.activation,
            global_num_experts=global_num_experts,
            expert_map=wave_map,
            a1q_scale=a1q_scale,
            a2_scale=fused_experts.a2_scale,
            workspace13=workspace13,
            workspace2=workspace2,
            expert_tokens_meta=expert_tokens_meta,
            apply_router_weight_on_input=False,
        )
        pair_outputs = workspace2.reshape(-1)[:required_pair_values].view(
            num_tokens * topk, hidden_size
        )
        selected = pair_outputs.index_select(0, pair_offsets.long()).clone()
        token_indices = torch.div(pair_offsets, topk, rounding_mode="floor")
        return NativeWavePayload(
            outputs=selected,
            pair_offsets=torch.empty(0, dtype=torch.long, device=output.device),
            token_indices=token_indices,
        )

    def combine(
        self,
        *,
        waves: Sequence[NativeWavePayload],
        topk_weights: torch.Tensor,
        pair_offsets: torch.Tensor,
        restore_shape: tuple[int, ...],
    ) -> torch.Tensor:
        if not waves:
            raise NativeCombineError("cannot combine an empty wave list")
        total_pairs = int(pair_offsets.numel())
        hidden = int(restore_shape[-1])
        fused = torch.empty(
            (total_pairs, hidden),
            dtype=waves[0].outputs.dtype,
            device=waves[0].outputs.device,
        )
        for wave in waves:
            fused.index_copy_(0, wave.pair_offsets, wave.outputs)
        dummy_ids = torch.zeros_like(topk_weights, dtype=torch.long)
        return self._reducer.apply(
            output=None,
            fused_expert_output=fused,
            topk_weights=topk_weights,
            topk_ids=dummy_ids,
            # Triton's second expert GEMM already applied the original router
            # weights before we copied the pair workspace.
            apply_router_weight_on_input=True,
        ).reshape(restore_shape)


class SpyMoeSeam(FunctionalMoeSeam):
    """Test seam that counts expert and combine calls."""

    def __init__(self, combine_fn: Callable[..., torch.Tensor]):
        self.run_calls = 0
        self.combine_calls = 0
        super().__init__(combine_fn=combine_fn)

    def run_expert_mlp(self, **kwargs) -> NativeWavePayload:
        self.run_calls += 1
        return super().run_expert_mlp(**kwargs)

    def combine(self, **kwargs) -> torch.Tensor:
        self.combine_calls += 1
        return super().combine(**kwargs)
