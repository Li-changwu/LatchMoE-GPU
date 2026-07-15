import pytest
import torch
from torch import nn

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader


pytestmark = pytest.mark.cuda


class _LazyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty((4, 4, 2), dtype=torch.bfloat16), requires_grad=False
        )
        self.w2_weight = nn.Parameter(
            torch.empty((4, 2, 2), dtype=torch.bfloat16), requires_grad=False
        )
        self.w13_weight.weight_loader = lambda *args, **kwargs: True
        self.w2_weight.weight_loader = lambda *args, **kwargs: True


class _LazyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _LazyExperts()


class _LazyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _LazyMlp()
        self.non_expert = nn.Parameter(
            torch.empty((2, 2), dtype=torch.bfloat16), requires_grad=False
        )
        self.register_buffer("state_buffer", torch.empty(2, dtype=torch.bfloat16))


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
def test_selected_lazy_layer_constructs_experts_off_device(tiny_manifest):
    observed_devices: list[str] = []

    def modules():
        module = _LazyDecoder()
        observed_devices.append(module.mlp.experts.w13_weight.device.type)
        yield module

    offloader = CudaSEWOffloader(tiny_manifest)
    with torch.device("cuda"):
        (module,) = offloader.wrap_modules(modules())

    assert observed_devices == ["cpu"]
    assert module.mlp.experts.w13_weight.device.type == "cpu"
    assert module.mlp.experts.w2_weight.device.type == "cpu"
    assert module.non_expert.device.type == "cuda"
    assert module.state_buffer.device.type == "cuda"
    assert offloader.host_store.bindings[0].original_device.type == "cuda"

    offloader.post_init()

    assert offloader.runtimes[0].slot_w13.device.type == "cuda"


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


def test_offloader_connects_profile_writer_from_environment(
    monkeypatch, tmp_path, tiny_manifest
):
    profile_path = tmp_path / "profile.jsonl"
    monkeypatch.setenv("VLLM_LATCHMOE_PROFILE_PATH", str(profile_path))

    offloader = CudaSEWOffloader(tiny_manifest, pin_memory=False)

    assert offloader.event_writer is not None
    offloader.event_writer.write("test_event", layer_id=0)
    assert '"event":"test_event"' in profile_path.read_text()


def test_latchmoe_post_init_rejects_missing_manifest_layer(tiny_manifest):
    offloader = CudaSEWOffloader(tiny_manifest, pin_memory=False)
    offloader.wrap_modules(iter(()))

    with pytest.raises(RuntimeError, match="missing manifest layers"):
        offloader.post_init()
