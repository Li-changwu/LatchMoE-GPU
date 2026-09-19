from types import SimpleNamespace

import torch

from vllm_latchmoe_cuda.moe_seam import VllmModularMoeSeam


class TritonExperts:
    a2_scale = None

    def __init__(self):
        self.seen_shapes = []

    def moe_problem_size(self, hidden, w1, w2, topk_ids):
        del w2
        return w1.shape[0], hidden.shape[0], w1.shape[1], hidden.shape[1], topk_ids.shape[1]

    def apply(
        self,
        *,
        output,
        hidden_states,
        topk_weights,
        topk_ids,
        expert_map,
        workspace2,
        **kwargs,
    ):
        del hidden_states, kwargs
        self.seen_shapes.append(tuple(topk_ids.shape))
        pair_values = topk_ids.numel() * output.shape[-1]
        pairs = workspace2.reshape(-1)[:pair_values].view(
            topk_ids.numel(), output.shape[-1]
        )
        pairs.fill_(-99)
        flat_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        for offset, logical_id in enumerate(flat_ids.tolist()):
            if int(expert_map[logical_id]) >= 0:
                pairs[offset].fill_(float(logical_id) + float(flat_weights[offset]))
        output.zero_()


class _Kernel:
    def __init__(self, fused_experts):
        self.fused_experts = fused_experts
        self.workspace13 = torch.empty(32)
        self.workspace2 = torch.empty(32)
        self.output = torch.empty((2, 2))

    def _prepare(self, hidden_states, topk_weights, topk_ids, **kwargs):
        del kwargs
        return hidden_states, None, None, topk_ids, topk_weights

    def _allocate_buffers(self, *args):
        del args
        return self.workspace13, self.workspace2, self.output


def test_vllm_wave_preserves_native_shape_offsets_and_workspace_payload():
    fused_experts = TritonExperts()
    kernel_impl = _Kernel(fused_experts)
    kernel = SimpleNamespace(impl=kernel_impl)
    experts = SimpleNamespace(
        quant_method=SimpleNamespace(kernel=kernel),
        apply_router_weight_on_input=False,
        activation="silu",
    )
    runtime = SimpleNamespace(
        num_experts=4,
        log2phy=torch.tensor([0, 1, -1, -1], dtype=torch.long),
    )
    seam = VllmModularMoeSeam(experts_module=experts, runtime=runtime)
    topk_ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.long)
    topk_weights = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    hidden = torch.ones((2, 2))
    slot_w13 = torch.empty((2, 4, 2))
    slot_w2 = torch.empty((2, 2, 2))

    first = seam.run_expert_mlp(
        hidden_states=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        pair_offsets=torch.tensor([0, 1]),
        logical_ids=torch.tensor([0, 1]),
        physical_ids=torch.tensor([0, 1]),
        slot_w13=slot_w13,
        slot_w2=slot_w2,
    )
    first_copy = first.outputs.clone()

    runtime.log2phy.copy_(torch.tensor([-1, -1, 0, 1]))
    second = seam.run_expert_mlp(
        hidden_states=hidden,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        pair_offsets=torch.tensor([2, 3]),
        logical_ids=torch.tensor([2, 3]),
        physical_ids=torch.tensor([0, 1]),
        slot_w13=slot_w13,
        slot_w2=slot_w2,
    )

    assert fused_experts.seen_shapes == [(2, 2), (2, 2)]
    assert first.token_indices.tolist() == [0, 0]
    assert second.token_indices.tolist() == [1, 1]
    torch.testing.assert_close(first.outputs, first_copy)
    torch.testing.assert_close(first.outputs[:, 0], torch.tensor([0.1, 1.2]))
    torch.testing.assert_close(second.outputs[:, 0], torch.tensor([2.3, 3.4]))
