from __future__ import annotations

import torch


RAW_TOPK_CPU_THRESHOLD = 512


def active_experts_from_topk(topk_ids: torch.Tensor) -> tuple[int, ...]:
    """Return sorted expert ids with a low-launch-overhead small-route path."""
    flat_ids = topk_ids.detach().reshape(-1)
    if flat_ids.numel() <= RAW_TOPK_CPU_THRESHOLD:
        values = flat_ids.to("cpu", non_blocking=False).tolist()
        return tuple(sorted({int(value) for value in values}))
    return tuple(
        int(value) for value in torch.unique(flat_ids, sorted=True).cpu().tolist()
    )
