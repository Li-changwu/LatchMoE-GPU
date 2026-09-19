import gc
import weakref

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.utils import device_loading_context
from vllm.v1.worker.workspace import init_workspace_manager

from vllm_latchmoe_cuda.graph_ops import (
    graph_fused_moe_compute,
    register_graph_runtime,
)
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.moe_seam import SpyMoeSeam
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


def _native_combine(*, waves, topk_weights, pair_offsets, restore_shape):
    output = torch.zeros(restore_shape, device=pair_offsets.device, dtype=torch.float32)
    weights = topk_weights.reshape(-1).float()
    for wave in waves:
        values = wave.outputs.float() * weights.index_select(
            0, wave.pair_offsets
        ).unsqueeze(-1)
        output.scatter_add_(
            0, wave.token_indices[:, None].expand_as(values), values
        )
    return output.to(dtype=torch.bfloat16)


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


def test_finite_slot_graph_replays_dynamic_map_and_overflow_uses_main_cache(
    tiny_manifest,
):
    init_workspace_manager(torch.device("cuda"))
    module = _Decoder()
    experts = module.mlp.experts
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(47)
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    full_w13 = offloader.host_store.tensor_view(0, "w13_weight").to("cuda")
    full_w2 = offloader.host_store.tensor_view(0, "w2_weight").to("cuda")
    with device_loading_context(experts, torch.device("cuda")):
        experts.quant_method.process_weights_after_loading(experts)
    offloader.post_init()
    runtime = offloader.runtimes[0]
    runtime_id = register_graph_runtime(runtime, experts)

    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32, device="cuda"
    )
    runtime.stage_sync((0, 1))
    for _ in range(3):
        graph_fused_moe_compute(runtime, runtime_id, hidden, topk_weights, topk_ids)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = graph_fused_moe_compute(
            runtime, runtime_id, hidden, topk_weights, topk_ids
        )
    torch.cuda.synchronize()

    replacement_ids = torch.tensor([[2, 3], [3, 2]], dtype=torch.int64, device="cuda")
    topk_ids.copy_(replacement_ids)
    runtime.stage_sync((2, 3))
    expected = _capturable_weights_moe(
        full_w13, full_w2, hidden, replacement_ids, topk_weights
    )
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured.float(), expected.float(), rtol=2e-2, atol=2e-2)

    assert not hasattr(runtime, "stage_pool")
    assert runtime.main_cache.slot_w13.data_ptr() == runtime.slot_w13.data_ptr()
    runtime.prepare_compute_async((2, 3))
    pending_output = graph_fused_moe_compute(
        runtime, runtime_id, hidden, topk_weights, topk_ids
    )
    runtime.finish_compute_async()
    overflow_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
    experts._latchmoe_seam = SpyMoeSeam(_native_combine)
    runtime.production_plan = True
    runtime.prepare_graph_compute((0, 1, 2, 3))
    overflow = graph_fused_moe_compute(
        runtime, runtime_id, hidden, topk_weights, overflow_ids
    )
    runtime.finish_graph_compute()
    overflow_expected = _capturable_weights_moe(
        full_w13, full_w2, hidden, overflow_ids, topk_weights
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(
        overflow.float(), overflow_expected.float(), rtol=2e-2, atol=2e-2
    )
    torch.testing.assert_close(
        pending_output.float(), expected.float(), rtol=2e-2, atol=2e-2
    )
    assert experts._latchmoe_seam.run_calls == 2
    assert experts._latchmoe_seam.combine_calls == 1
    assert torch.count_nonzero(runtime.log2phy == -1).item() == (
        runtime.num_experts - runtime.num_slots
    )
