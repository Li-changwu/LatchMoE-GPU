import torch

from vllm_latchmoe_cuda.host_store import PinnedHostStore
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.residency_plan import build_residency_plan


def _four_layer_plan():
    return build_residency_plan(
        191 / (1 << 30),
        {"model_type": "synthetic_moe", "moe_layer_ids": [0, 1, 2, 3], "num_experts": 4},
        max_capture_size=1,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=tuple(
            {"layer_id": layer, "routed_expert_bytes": 96} for layer in range(4)
        ),
    )


def test_plan_path_binds_only_selected_layers(tiny_decoder_factory):
    plan = _four_layer_plan()
    modules = tuple(tiny_decoder_factory("cpu") for _ in range(4))
    original = tuple(module.mlp.experts.w13_weight.data_ptr() for module in modules)
    offloader = CudaSEWOffloader(plan=plan, pin_memory=False)

    wrapped = offloader.wrap_modules(iter(modules))

    assert wrapped == list(modules)
    assert plan.offloaded_layer_ids == (1, 3)
    assert {binding.layer_id for binding in offloader.host_store.bindings} == {1, 3}
    assert modules[0].mlp.experts.w13_weight.data_ptr() == original[0]
    assert modules[2].mlp.experts.w13_weight.data_ptr() == original[2]
    assert modules[1].mlp.experts.w13_weight.device.type == "cpu"
    assert modules[3].mlp.experts.w13_weight.device.type == "cpu"


def test_dynamic_store_records_actual_layout(tiny_decoder_factory):
    parameter = tiny_decoder_factory("cpu").mlp.experts.w13_weight
    store = PinnedHostStore(pin_memory=False)
    binding = store.bind_parameter(7, "w13_weight", parameter)

    assert binding.shape == tuple(parameter.shape)
    assert binding.stride == tuple(parameter.stride())
    assert binding.dtype == "bfloat16"
    assert binding.nbytes == parameter.numel() * parameter.element_size()

