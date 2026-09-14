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
        physical_ids: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
    ) -> NativeWavePayload:
        ids = physical_ids.reshape(-1).long()
        weights = slot_w13.index_select(0, ids).float()
        hidden = hidden_states.float()
        gate_up = torch.einsum("pih,ph->pi", weights, hidden)
        gate, up = gate_up.chunk(2, dim=-1)
        intermediate = self.activation(gate) * up
        down = slot_w2.index_select(0, ids).float()
        outputs = torch.einsum("poi,pi->po", down, intermediate)
        return NativeWavePayload(
            outputs=outputs,
            pair_offsets=torch.empty(0, dtype=torch.long, device=outputs.device),
            token_indices=torch.arange(
                outputs.shape[0], dtype=torch.long, device=outputs.device
            ),
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
    modular kernel does expose the two primitives we need, however: applying
    the native expert kernel and ``TopKWeightAndReduceContiguous``.  Each wave
    is submitted as a top-1 batch (so the native kernel returns one output per
    routed pair), then the native reducer combines all wave outputs once.
    """

    def __init__(self, *, experts_module, runtime):
        self.experts_module = experts_module
        self.runtime = runtime
        from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
            TopKWeightAndReduceContiguous,
        )

        self._reducer = TopKWeightAndReduceContiguous()

    def _logical_ids(self, physical_ids: torch.Tensor) -> torch.Tensor:
        mapping = self.runtime.log2phy
        logical = torch.full_like(physical_ids, -1)
        for expert_id in range(self.runtime.num_experts):
            logical = torch.where(
                mapping[expert_id] == physical_ids,
                torch.full_like(logical, expert_id),
                logical,
            )
        if bool(torch.any(logical < 0)):
            raise NativeCombineError("Main Cache physical expert has no logical owner")
        return logical

    def run_expert_mlp(
        self,
        *,
        hidden_states: torch.Tensor,
        physical_ids: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
    ) -> NativeWavePayload:
        quant_method = getattr(self.experts_module, "quant_method", None)
        apply = getattr(quant_method, "apply", None)
        if not callable(apply):
            raise NativeCombineError("vLLM modular quant method has no apply primitive")
        logical_ids = self._logical_ids(physical_ids).reshape(-1, 1)
        topk_weights = torch.ones(
            logical_ids.shape,
            dtype=torch.float32,
            device=hidden_states.device,
        )
        output = apply(
            layer=self.experts_module,
            x=hidden_states,
            topk_weights=topk_weights,
            topk_ids=logical_ids,
            shared_experts_input=hidden_states,
        )
        if isinstance(output, tuple):
            output = output[-1]
        if not isinstance(output, torch.Tensor) or output.ndim != 2:
            raise NativeCombineError("vLLM modular expert primitive returned an invalid shape")
        return NativeWavePayload(
            outputs=output,
            pair_offsets=torch.empty(0, dtype=torch.long, device=output.device),
            token_indices=torch.arange(output.shape[0], device=output.device),
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
            apply_router_weight_on_input=False,
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
