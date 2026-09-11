from types import SimpleNamespace

import pytest

from vllm_latchmoe_cuda.capabilities import (
    CapabilityDescriptor,
    describe_capabilities,
    validate_capabilities,
)
from vllm_latchmoe_cuda.errors import CapabilityError


def _descriptor(**overrides):
    values = dict(
        model_family="qwen3_moe",
        router_owner="router.Router",
        shared_expert_representation="none",
        dtype="bfloat16",
        tensor_parallel_size=1,
        expert_parallel=False,
        quant_method="vllm.Quant",
        kernel_mode="modular",
        output_abi="tensor",
        combine_owner="vllm.native",
        graph_mode="piecewise",
    )
    values.update(overrides)
    return CapabilityDescriptor(**values)


def test_qualified_routed_only_tuple_is_accepted():
    validate_capabilities(_descriptor())


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_family", "glm4_moe_lite"),
        ("dtype", "float16"),
        ("tensor_parallel_size", 2),
        ("expert_parallel", True),
        ("shared_expert_representation", "external"),
        ("kernel_mode", "monolithic"),
    ],
)
def test_unqualified_tuple_fails_closed(field, value):
    with pytest.raises(CapabilityError):
        validate_capabilities(_descriptor(**{field: value}))


def test_describe_capabilities_records_router_kernel_and_shared_representation():
    module = SimpleNamespace(
        router=SimpleNamespace(),
        quant_method=SimpleNamespace(is_monolithic=False, moe_kernel=object()),
        _shared_experts=None,
    )
    descriptor = describe_capabilities(module)

    assert descriptor.router_owner.endswith("SimpleNamespace")
    assert descriptor.kernel_mode == "modular"
    assert descriptor.shared_expert_representation == "none"
    assert descriptor.to_jsonable()["combine_owner"] == "vllm.native"

