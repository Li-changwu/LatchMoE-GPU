from __future__ import annotations

import torch
import torch.nn.functional as F

from .runtime import CudaLayerRuntime


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
        output = torch.zeros(
            (hidden_states.shape[0], runtime.slot_w2.shape[1]),
            dtype=torch.float32,
            device=hidden_states.device,
        )
        for token_index in range(hidden_states.shape[0]):
            x = hidden_states[token_index].float()
            for topk_position in range(topk_ids.shape[1]):
                slot_id = int(physical_ids[token_index, topk_position].item())
                gate_up = torch.matmul(runtime.slot_w13[slot_id].float(), x)
                gate, up = gate_up.chunk(2, dim=0)
                expert_output = torch.matmul(
                    runtime.slot_w2[slot_id].float(), F.silu(gate) * up
                )
                output[token_index].add_(
                    expert_output * topk_weights[token_index, topk_position]
                )
        return output
    finally:
        runtime.end_compute(handle)

