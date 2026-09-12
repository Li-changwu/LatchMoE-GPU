import json

import pytest
import torch
from torch import nn

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import capturable_slot_moe


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


class _Router:
    def select_experts(self, *, hidden_states, router_logits):
        values, ids = torch.topk(router_logits, k=2, dim=-1)
        return torch.softmax(values.float(), dim=-1), ids


class _Quant:
    is_monolithic = False

    def __init__(self, runtime):
        self.runtime = runtime

    def apply(self, *, layer, x, topk_weights, topk_ids, shared_experts_input):
        physical = layer.expert_map[topk_ids].long()
        return capturable_slot_moe(self.runtime, x, physical, topk_weights)


class _ExternalShared(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.full((2,), 2.0, device="cuda"))
        self.calls = 0

    def forward(self, hidden_states):
        self.calls += 1
        return hidden_states * self.weight


def test_external_shared_expert_is_resident_and_called_once(
    tiny_manifest, tiny_decoder_factory, monkeypatch, tmp_path
):
    monkeypatch.setenv("VLLM_LATCHMOE_GRAPH_MODE", "eager")
    profile_path = tmp_path / "shared.jsonl"
    monkeypatch.setenv("VLLM_LATCHMOE_PROFILE_PATH", str(profile_path))

    module = tiny_decoder_factory("cuda")
    shared = _ExternalShared()
    experts = module.mlp.experts
    experts._shared_experts = shared
    experts.returns_shared_experts = True
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        offloader.host_store.tensor_view(0, name).normal_()
    offloader.post_init()

    runtime = offloader.runtimes[0]
    runtime._shared_seam = None
    experts.router = _Router()
    experts.quant_method = _Quant(runtime)
    # post_init installs the adapter only when router/quant are already set;
    # install once here to keep the fixture's load order explicit.
    from vllm_latchmoe_cuda.runner_adapter import install_vllm_forward_adapter

    install_vllm_forward_adapter(experts, runtime)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    logits = torch.tensor(
        [[4.0, 3.0, -1.0, -2.0], [3.0, 4.0, -2.0, -1.0]], device="cuda"
    )

    shared_output, routed_output = experts(hidden, logits)
    shared_output_again, _ = experts(hidden, logits)

    assert shared.calls == 2
    torch.testing.assert_close(shared_output.float(), (hidden * 2).float())
    torch.testing.assert_close(shared_output_again.float(), shared_output.float())
    assert routed_output.shape == hidden.shape
    assert all(binding.name in {"w13_weight", "w2_weight"} for binding in offloader.host_store.bindings)
    assert shared.weight.device.type == "cuda"
    assert runtime.resident_shared_weight_bytes == shared.weight.numel() * shared.weight.element_size()
    assert runtime.dynamic_slot_bytes == runtime.slot_w13.numel() * runtime.slot_w13.element_size() + runtime.slot_w2.numel() * runtime.slot_w2.element_size()

    offloader.close()
    events = [json.loads(line) for line in profile_path.read_text().splitlines()]
    ledger = next(event for event in events if event["event"] == "residency_ledger")
    assert ledger["resident_shared_weight_bytes"] == runtime.resident_shared_weight_bytes
    assert ledger["dynamic_slot_bytes"] == runtime.dynamic_slot_bytes
    assert ledger["host_routed_expert_bytes"] > 0
