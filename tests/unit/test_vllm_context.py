from types import SimpleNamespace

import pytest

from vllm_latchmoe_cuda.vllm_context import advance_moe_layer_index


def _install_context(monkeypatch, context) -> None:
    import vllm.forward_context as forward_context

    monkeypatch.setattr(forward_context, "is_forward_context_available", lambda: True)
    monkeypatch.setattr(forward_context, "get_forward_context", lambda: context)


def test_advance_moe_layer_index_matches_vllm_order(monkeypatch):
    context = SimpleNamespace(
        all_moe_layers=["model.layers.0.mlp.experts"], moe_layer_index=0
    )
    experts = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
    _install_context(monkeypatch, context)

    advance_moe_layer_index(experts)

    assert context.moe_layer_index == 1


def test_advance_moe_layer_index_skips_unobserved_registered_layers(monkeypatch):
    context = SimpleNamespace(
        all_moe_layers=[
            "model.layers.0.mlp.experts",
            "model.layers.1.mlp.experts",
            "model.layers.2.mlp.experts",
            "model.layers.3.mlp.experts",
        ],
        moe_layer_index=0,
    )
    _install_context(monkeypatch, context)
    advance_moe_layer_index(
        SimpleNamespace(layer_name="model.layers.3.mlp.experts")
    )
    assert context.moe_layer_index == 4


def test_advance_moe_layer_index_rejects_layer_order_mismatch(monkeypatch):
    context = SimpleNamespace(
        all_moe_layers=["model.layers.1.mlp.experts"], moe_layer_index=0
    )
    experts = SimpleNamespace(layer_name="model.layers.0.mlp.experts")
    _install_context(monkeypatch, context)

    with pytest.raises(RuntimeError, match="layer order differs"):
        advance_moe_layer_index(experts)


def test_advance_moe_layer_index_rejects_extra_calls(monkeypatch):
    context = SimpleNamespace(all_moe_layers=[], moe_layer_index=0)
    _install_context(monkeypatch, context)

    with pytest.raises(RuntimeError, match="more routed-expert calls"):
        advance_moe_layer_index(SimpleNamespace(layer_name="layer"))


def test_advance_moe_layer_index_ignores_context_without_layer_order(monkeypatch):
    context = SimpleNamespace(all_moe_layers=None, moe_layer_index=0)
    _install_context(monkeypatch, context)

    advance_moe_layer_index(SimpleNamespace(layer_name="layer"))

    assert context.moe_layer_index == 0
