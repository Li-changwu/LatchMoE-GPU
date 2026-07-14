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
    assert not hasattr(offloader, "_cpu_tensors")
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

    observed = parameter.detach().cpu()
    assert torch.equal(observed, loaded.to(dtype=torch.bfloat16))
    assert getattr(parameter, "_vllm_is_uva_offloaded", False)


@pytest.mark.parametrize(
    ("pin_memory", "use_uva"), [(False, True), (True, False), (False, False)]
)
def test_manifest_uva_fails_closed_without_pinned_uva(
    tiny_manifest, pin_memory, use_uva
):
    with pytest.raises(RuntimeError, match="pinned UVA"):
        ManifestUVAOffloader(
            tiny_manifest,
            pin_memory=pin_memory,
            use_uva=use_uva,
        )


def test_manifest_uva_rejects_missing_manifest_layer(tiny_manifest):
    offloader = ManifestUVAOffloader(
        tiny_manifest,
        pin_memory=True,
        use_uva=True,
    )

    with pytest.raises(RuntimeError, match="missing manifest layers"):
        offloader.wrap_modules(iter(()))
