from __future__ import annotations

from torch import nn


def advance_moe_layer_index(experts_module: nn.Module) -> None:
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if not is_forward_context_available():
        return
    context = get_forward_context()
    all_moe_layers = context.all_moe_layers
    if all_moe_layers is None:
        return
    index = context.moe_layer_index
    if index >= len(all_moe_layers):
        raise RuntimeError(
            "LatchMoE encountered more routed-expert calls than vLLM registered"
        )
    expected = all_moe_layers[index]
    actual = getattr(experts_module, "layer_name", None)
    if actual is not None and expected != actual:
        raise RuntimeError(
            "LatchMoE/vLLM MoE layer order differs: "
            f"index={index}, expected={expected!r}, actual={actual!r}"
        )
    context.moe_layer_index += 1
