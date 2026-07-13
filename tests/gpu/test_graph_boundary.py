import pytest
import torch

from vllm_latchmoe_cuda.errors import StagingDuringCaptureError
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import capturable_slot_moe
from vllm_latchmoe_cuda.split_ops import eager_stage_and_map


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _runtime(tiny_manifest, tiny_decoder_factory):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    torch.manual_seed(17)
    for parameter in module.mlp.experts.parameters():
        parameter.data.copy_(torch.randn_like(parameter, device="cpu"))
    offloader.post_init()
    return offloader.runtimes[0]


def test_dynamic_staging_rejects_capture(monkeypatch, tiny_manifest, tiny_decoder_factory):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(StagingDuringCaptureError, match="layer=0"):
        runtime.stage_async((0, 1))


def test_eager_split_produces_a_dynamo_graph_break(tiny_manifest, tiny_decoder_factory):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    ids = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")

    def split_function(x, topk_ids):
        before = x + 1
        eager_stage_and_map(runtime, topk_ids)
        return before * 2

    explanation = torch._dynamo.explain(split_function)(
        torch.ones((1,), device="cuda"), ids
    )

    assert explanation.graph_count >= 1
    assert explanation.graph_break_count >= 1


def test_slot_compute_can_be_captured_and_replayed(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    topk_ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="cuda")
    topk_weights = torch.tensor(
        [[0.6, 0.4], [0.25, 0.75]], dtype=torch.float32, device="cuda"
    )
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    runtime.stage_sync((0, 1))
    physical_ids = runtime.log2phy[topk_ids].long()
    for _ in range(3):
        expected = capturable_slot_moe(runtime, hidden, physical_ids, topk_weights)
    torch.cuda.synchronize()
    pointers = runtime.data_ptrs()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = capturable_slot_moe(runtime, hidden, physical_ids, topk_weights)
    graph.replay()
    torch.cuda.synchronize()

    torch.testing.assert_close(captured, expected, rtol=2e-2, atol=2e-2)
    assert runtime.data_ptrs() == pointers

