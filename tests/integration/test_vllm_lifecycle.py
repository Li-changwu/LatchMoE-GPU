import gc
import weakref

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import (
    _capturable_weights_moe,
    capturable_slot_moe,
    execute_exact_waves,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        with set_current_vllm_config(VllmConfig()):
            with torch.device("cuda"):
                self.experts = FusedMoE(
                    num_experts=4,
                    top_k=2,
                    hidden_size=2,
                    intermediate_size=2,
                    params_dtype=torch.bfloat16,
                    tp_size=1,
                    ep_size=1,
                    dp_size=1,
                    pcp_size=1,
                    prefix="mlp.experts",
                )


class _Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _Mlp()


def test_selected_real_fused_moe_postprocess_never_materializes_full_weights_on_cuda(
    tiny_manifest,
):
    module = _Decoder()
    experts = module.mlp.experts
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    old_w13 = weakref.ref(experts.w13_weight)
    assert experts.w13_weight.device.type == "cpu"
    assert "mlp.experts.w13_weight" in dict(module.named_parameters())
    assert "w13_weight" not in dict(experts.named_parameters())

    with device_loading_context(experts, torch.device("cuda")):
        assert experts.w13_weight.device.type == "cpu"
        assert experts.w2_weight.device.type == "cpu"
        experts.quant_method.process_weights_after_loading(experts)
        assert experts.w13_weight.device.type == "cpu"
        assert experts.w2_weight.device.type == "cpu"

    gc.collect()
    assert old_w13() is None
    offloader.post_init()

    runtime = offloader.runtimes[0]
    assert tuple(experts.w13_weight.shape) == (tiny_manifest.num_slots, 4, 2)
    assert experts.w13_weight.data_ptr() == runtime.slot_w13.data_ptr()
    assert "w13_weight" in dict(experts.named_parameters())


def test_real_triton_quant_method_uses_staged_slots_and_expert_map(tiny_manifest):
    init_workspace_manager(torch.device("cuda"))
    module = _Decoder()
    experts = module.mlp.experts
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(43)
    offloader.host_store.tensor_view(0, "w13_weight").copy_(
        torch.randn_like(offloader.host_store.tensor_view(0, "w13_weight"))
    )
    offloader.host_store.tensor_view(0, "w2_weight").copy_(
        torch.randn_like(offloader.host_store.tensor_view(0, "w2_weight"))
    )
    full_w13 = offloader.host_store.tensor_view(0, "w13_weight").to("cuda")
    full_w2 = offloader.host_store.tensor_view(0, "w2_weight").to("cuda")
    with device_loading_context(experts, torch.device("cuda")):
        experts.quant_method.process_weights_after_loading(experts)
    offloader.post_init()
    runtime = offloader.runtimes[0]
    runtime.stage_sync((0, 1))
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32, device="cuda"
    )
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")

    actual = experts.quant_method.apply(
        layer=experts,
        x=hidden,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        shared_experts_input=hidden,
    )
    expected = capturable_slot_moe(
        runtime, hidden, runtime.log2phy[topk_ids].long(), topk_weights
    )

    torch.testing.assert_close(actual.float(), expected.float(), rtol=2e-2, atol=2e-2)

    wave_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
    wave_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32, device="cuda"
    )

    def kernel_callback(pair_hidden, logical_ids, pair_weights):
        return experts.quant_method.apply(
            layer=experts,
            x=pair_hidden,
            topk_weights=pair_weights,
            topk_ids=logical_ids,
            shared_experts_input=pair_hidden,
        )

    wave_expected = _capturable_weights_moe(
        full_w13, full_w2, hidden, wave_ids, wave_weights
    )
    wave_actual = execute_exact_waves(
        runtime,
        hidden.clone(),
        wave_ids,
        wave_weights,
        kernel_callback=kernel_callback,
    )

    torch.testing.assert_close(
        wave_actual.float(), wave_expected.float(), rtol=2e-2, atol=2e-2
    )
