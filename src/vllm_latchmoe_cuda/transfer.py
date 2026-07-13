from __future__ import annotations

import torch


def copy_expert_sync(
    *,
    host_w13: torch.Tensor,
    host_w2: torch.Tensor,
    expert_id: int,
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    slot_id: int,
) -> None:
    slot_w13[slot_id].copy_(host_w13[expert_id], non_blocking=False)
    slot_w2[slot_id].copy_(host_w2[expert_id], non_blocking=False)

