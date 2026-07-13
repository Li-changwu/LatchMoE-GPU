import pytest
import torch

from vllm_latchmoe_cuda.uva import ManifestUVAOffloader


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_uva_selects_only_manifest_parameters(tiny_manifest, tiny_decoder_factory):
    selected = tiny_decoder_factory("cuda")
    untouched = tiny_decoder_factory("cuda")
    untouched_ptr = untouched.mlp.experts.w13_weight.data_ptr()
    offloader = ManifestUVAOffloader(tiny_manifest)

    modules = offloader.wrap_modules(iter((selected, untouched)))

    assert modules == [selected, untouched]
    assert offloader.offloaded_parameter_names == {
        "model.layers.0.mlp.experts.w13_weight",
        "model.layers.0.mlp.experts.w2_weight",
    }
    assert len(offloader.cpu_backing_tensors) == 2
    assert all(tensor.is_pinned() for tensor in offloader.cpu_backing_tensors)
    assert untouched.mlp.experts.w13_weight.data_ptr() == untouched_ptr


def test_uva_view_and_cpu_backing_observe_same_values(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = ManifestUVAOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    parameter = module.mlp.experts.w13_weight
    loaded = torch.arange(parameter.numel(), dtype=torch.float32).view_as(parameter)

    assert parameter.weight_loader(parameter, loaded)
    torch.cuda.synchronize()

    backing = offloader.cpu_tensor(0, "w13_weight")
    assert torch.equal(backing, loaded.to(dtype=torch.bfloat16))
    assert getattr(parameter, "_vllm_is_uva_offloaded", False)
