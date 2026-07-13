import pytest
import torch

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader


pytestmark = pytest.mark.cuda


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_wrap_modules_redirects_before_checkpoint_load(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    w13_parameter = module.mlp.experts.w13_weight
    w2_parameter = module.mlp.experts.w2_weight
    w13_loader = w13_parameter.weight_loader
    offloader = CudaSEWOffloader(tiny_manifest)

    wrapped = offloader.wrap_modules(iter((module,)))

    assert wrapped == [module]
    assert module.mlp.experts.w13_weight is w13_parameter
    assert module.mlp.experts.w2_weight is w2_parameter
    assert w13_parameter.weight_loader is w13_loader
    assert w13_parameter.device.type == "cpu"
    assert w2_parameter.device.type == "cpu"
    assert w13_parameter.is_pinned()
    assert w2_parameter.is_pinned()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_checkpoint_loader_writes_directly_to_pinned_store(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    param = module.mlp.experts.w13_weight
    loaded = torch.arange(param.numel(), dtype=torch.float32).view_as(param)

    assert param.weight_loader(param, loaded)

    expected = loaded.to(dtype=torch.bfloat16)
    assert torch.equal(offloader.host_store.tensor_view(0, "w13_weight"), expected)
    assert offloader.bound_parameter_names == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }
