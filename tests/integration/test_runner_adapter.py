import pytest
import torch

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import (
    capturable_slot_moe,
    install_vllm_forward_adapter,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


class FakeRouter:
    def select_experts(self, *, hidden_states, router_logits):
        weights, ids = torch.topk(router_logits, k=2, dim=-1)
        return torch.softmax(weights.float(), dim=-1), ids


class FakeQuantMethod:
    is_monolithic = False

    def __init__(self, runtime):
        self.runtime = runtime
        self.calls = 0

    def apply(self, *, layer, x, topk_weights, topk_ids, shared_experts_input):
        self.calls += 1
        physical = layer.expert_map[topk_ids].long()
        return capturable_slot_moe(self.runtime, x, physical, topk_weights)


class RecordingWriter:
    def __init__(self):
        self.events = []

    def write(self, event, **fields):
        self.events.append({"event": event, **fields})


def test_instance_adapter_stages_between_router_and_original_quant_kernel(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    host_w13 = offloader.host_store.tensor_view(0, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(0, "w2_weight")
    torch.manual_seed(23)
    host_w13.copy_(torch.randn_like(host_w13))
    host_w2.copy_(torch.randn_like(host_w2))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    experts = module.mlp.experts
    experts.router = FakeRouter()
    experts.quant_method = FakeQuantMethod(runtime)
    experts._shared_experts = None
    install_vllm_forward_adapter(experts, runtime)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    router_logits = torch.tensor(
        [[4.0, 3.0, -1.0, -2.0], [3.0, 4.0, -2.0, -1.0]], device="cuda"
    )

    shared, output = experts(hidden, router_logits)
    shared_again, output_again = experts(hidden, router_logits)

    assert shared is None
    assert shared_again is None
    assert output.shape == hidden.shape
    torch.testing.assert_close(output_again, output, rtol=2e-2, atol=2e-2)
    assert experts.quant_method.calls == 2
    assert runtime.counters.snapshot()["slot_hit"] >= 2


def test_instance_adapter_routes_overflow_through_original_kernel_per_wave(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(29)
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    runtime.event_writer = RecordingWriter()
    experts = module.mlp.experts
    experts.router = FakeRouter()
    experts.quant_method = FakeQuantMethod(runtime)
    experts._shared_experts = None
    install_vllm_forward_adapter(experts, runtime)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    router_logits = torch.tensor(
        [[5.0, 4.0, -3.0, -4.0], [-4.0, -3.0, 5.0, 4.0]], device="cuda"
    )

    shared, output = experts(hidden, router_logits)

    assert shared is None
    assert output.shape == hidden.shape
    assert experts.quant_method.calls == 2
    assert runtime.last_wave_trace.pair_count == 4
    assert runtime.last_wave_trace.compute_order == (0, 1)
    assert runtime.event_writer.events == [
        {
            "event": "exact_waves",
            "layer_id": 0,
            "pair_count": 4,
            "wave_count": 2,
            "compute_order": [0, 1],
            "issue_order": [0, 1],
        }
    ]


def test_regular_request_reloads_after_overflow_overwrites_main_slots(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(37)
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    experts = module.mlp.experts
    experts.router = FakeRouter()
    experts.quant_method = FakeQuantMethod(runtime)
    experts._shared_experts = None
    install_vllm_forward_adapter(experts, runtime)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    regular_logits = torch.tensor(
        [[5.0, 4.0, -3.0, -4.0], [4.0, 5.0, -3.0, -4.0]], device="cuda"
    )
    overflow_logits = torch.tensor(
        [[5.0, 4.0, -3.0, -4.0], [-4.0, -3.0, 5.0, 4.0]], device="cuda"
    )

    _, before = experts(hidden, regular_logits)
    experts(hidden, overflow_logits)
    _, after = experts(hidden, regular_logits)

    torch.testing.assert_close(after, before, rtol=2e-2, atol=2e-2)
