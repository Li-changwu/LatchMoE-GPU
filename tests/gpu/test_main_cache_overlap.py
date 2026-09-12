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
    torch.manual_seed(101)
    for name in ("w13_weight", "w2_weight"):
        offloader.host_store.tensor_view(0, name).copy_(
            torch.randn_like(offloader.host_store.tensor_view(0, name))
        )
    offloader.post_init()
    return offloader.runtimes[0]


def test_partial_hit_prefetch_uses_only_idle_slot_and_records_window(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    hidden = torch.randn((1, 2), dtype=torch.bfloat16, device="cuda")
    weights = torch.tensor([[0.5, 0.5]], device="cuda")

    first_ids = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    second_ids = torch.tensor([[0, 2]], dtype=torch.int64, device="cuda")
    execute_main_cache_waves(runtime, hidden, first_ids, weights, overlap=False)
    output = execute_main_cache_waves(
        runtime, hidden, second_ids, weights, overlap=True
    )
    torch.cuda.synchronize()

    assert output.shape == hidden.shape
    assert len(runtime.overlap_records) == 1
    record = runtime.overlap_records[0]
    assert record["overlap_candidate"] is True
    assert isinstance(record["actual_overlap"], bool)
    assert record["h2d_bytes"] > 0
    assert record["protected_slot_ids"] == [0]
    assert runtime.main_cache.lease_for(2) is not None


def test_serial_and_overlap_have_same_result_for_partial_hit(
    tiny_manifest, tiny_decoder_factory
):
    serial_runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    overlap_runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    hidden = torch.randn((1, 2), dtype=torch.bfloat16, device="cuda")
    weights = torch.tensor([[0.5, 0.5]], device="cuda")
    first_ids = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    second_ids = torch.tensor([[0, 2]], dtype=torch.int64, device="cuda")

    execute_main_cache_waves(
        serial_runtime, hidden, first_ids, weights, overlap=False
    )
    serial = execute_main_cache_waves(
        serial_runtime, hidden, second_ids, weights, overlap=False
    )
    execute_main_cache_waves(
        overlap_runtime, hidden, first_ids, weights, overlap=False
    )
    overlap = execute_main_cache_waves(
        overlap_runtime, hidden, second_ids, weights, overlap=True
    )

    torch.testing.assert_close(overlap, serial, rtol=2e-2, atol=2e-2)
    assert overlap_runtime.last_wave_trace.pair_count == int(second_ids.numel())
    assert overlap_runtime.last_wave_trace.compute_order == (0, 1)
    ownership = {
        expert: overlap_runtime.main_cache.lease_for(expert).slot_id
        for expert in (0, 2)
    }
    assert ownership[0] != ownership[2]
