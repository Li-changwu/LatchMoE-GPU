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

