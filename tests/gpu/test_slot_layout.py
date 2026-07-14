import pytest
import torch

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _loaded_offloader(tiny_manifest, tiny_decoder_factory):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(7)
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    return module, offloader


def test_post_init_binds_parameters_to_persistent_cuda_slots(
    tiny_manifest, tiny_decoder_factory
):
    module, offloader = _loaded_offloader(tiny_manifest, tiny_decoder_factory)
    runtime = offloader.runtimes[0]

    assert module.mlp.experts.w13_weight is runtime.slot_w13_parameter
    assert module.mlp.experts.w2_weight is runtime.slot_w2_parameter
    assert module.mlp.experts.w13_weight.shape == (2, 4, 2)
    assert module.mlp.experts.w2_weight.shape == (2, 2, 2)
    assert module.mlp.experts.w13_weight.device.type == "cuda"
    assert runtime.log2phy.dtype == torch.int32
    assert runtime.log2phy.shape == (4,)


def test_slot_and_map_addresses_remain_stable(tiny_manifest, tiny_decoder_factory):
    _, offloader = _loaded_offloader(tiny_manifest, tiny_decoder_factory)
    runtime = offloader.runtimes[0]
    before = runtime.data_ptrs()

    runtime.stage_sync((1, 2))
    runtime.stage_sync((2, 3))

    assert runtime.data_ptrs() == before


def test_existing_none_expert_map_attribute_is_replaced_by_stable_buffer(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    module.mlp.experts._expert_map = None
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    offloader.post_init()

    runtime = offloader.runtimes[0]

    assert module.mlp.experts._buffers["_expert_map"] is runtime.log2phy
    assert module.mlp.experts._expert_map.data_ptr() == runtime.log2phy.data_ptr()
