import pytest
import torch

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_async_stage_compute_loop_has_bounded_allocator_growth(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for parameter in module.mlp.experts.parameters():
        parameter.data.copy_(torch.randn_like(parameter, device="cpu"))
    offloader.post_init()
    runtime = offloader.runtimes[0]

    for step in range(10):
        active = (step % 4, (step + 1) % 4)
        snapshot = runtime.stage_async(active)
        pending = runtime.end_compute_async(runtime.begin_compute(snapshot))
        runtime.wait_compute_done(pending)
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()

    for step in range(100):
        active = (step % 4, (step + 1) % 4)
        snapshot = runtime.stage_async(active)
        pending = runtime.end_compute_async(runtime.begin_compute(snapshot))
        runtime.wait_compute_done(pending)
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()

    assert after - before <= 1024 * 1024
