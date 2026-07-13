import pytest
import torch
import torch.nn.functional as F

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import eager_slot_moe


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _expert_mlp(x, w13, w2):
    gate_up = torch.matmul(w13.float(), x.float())
    gate, up = gate_up.chunk(2, dim=0)
    return torch.matmul(w2.float(), F.silu(gate) * up)


def test_sync_staged_output_matches_full_weight_reference(
    tiny_manifest, tiny_decoder_factory
):
    torch.manual_seed(11)
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    host_w13 = offloader.host_store.tensor_view(0, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(0, "w2_weight")
    host_w13.copy_(torch.randn_like(host_w13))
    host_w2.copy_(torch.randn_like(host_w2))
    full_w13 = host_w13.clone().to("cuda")
    full_w2 = host_w2.clone().to("cuda")
    offloader.post_init()
    runtime = offloader.runtimes[0]
    x = torch.randn((3, 2), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [1, 0], [0, 1]], device="cuda")
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8]],
        dtype=torch.float32,
        device="cuda",
    )
    expected = torch.zeros((3, 2), dtype=torch.float32, device="cuda")
    for token in range(x.shape[0]):
        for position in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, position])
            expected[token] += topk_weights[token, position] * _expert_mlp(
                x[token], full_w13[expert], full_w2[expert]
            )

    actual = eager_slot_moe(runtime, x, topk_ids, topk_weights)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
