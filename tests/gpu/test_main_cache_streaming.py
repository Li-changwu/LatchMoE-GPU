import pytest
import torch

from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import execute_main_cache_waves


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _runtime(tiny_manifest, tiny_decoder_factory):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    return offloader.runtimes[0]


def test_repeated_route_is_hit_first_and_reduces_h2d(tiny_manifest, tiny_decoder_factory):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [1, 0]], dtype=torch.int64, device="cuda")
    weights = torch.full((2, 2), 0.5, dtype=torch.float32, device="cuda")

    execute_main_cache_waves(runtime, hidden, ids, weights)
    first = runtime.counters.snapshot().get("h2d_bytes", 0)
    execute_main_cache_waves(runtime, hidden, ids, weights)
    second = runtime.counters.snapshot().get("h2d_bytes", 0) - first

    assert first > 0
    assert second == 0
    assert runtime.last_wave_trace.pair_count == ids.numel()


def test_union_over_capacity_replaces_in_same_main_cache(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    hidden = torch.randn((2, 2), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
    weights = torch.full((2, 2), 0.5, dtype=torch.float32, device="cuda")

    execute_main_cache_waves(runtime, hidden, ids, weights)
    torch.cuda.synchronize()

    assert runtime.main_cache.slot_w13.data_ptr() == runtime.slot_w13.data_ptr()
    assert runtime.main_cache.slot_w2.data_ptr() == runtime.slot_w2.data_ptr()
    assert runtime.last_wave_trace.compute_order == (0, 1)

