import pytest
import torch

from vllm_latchmoe_cuda.moe_seam import NativeWavePayload
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


class _DelayedSlotReadSeam:
    def run_expert_mlp(
        self,
        *,
        hidden_states,
        topk_ids,
        topk_weights,
        pair_offsets,
        logical_ids,
        physical_ids,
        slot_w13,
        slot_w2,
    ):
        del hidden_states, topk_ids, topk_weights, logical_ids, slot_w2
        # Keep the first wave reading its slots while the host starts planning
        # the second wave.  Missing stream ordering deterministically lets the
        # transfer stream overwrite these values before this read.
        torch.cuda._sleep(50_000_000)
        outputs = slot_w13.index_select(0, physical_ids.long())[:, 0, :].clone()
        return NativeWavePayload(
            outputs=outputs,
            pair_offsets=pair_offsets,
            token_indices=torch.empty_like(pair_offsets),
        )

    def combine(self, *, waves, topk_weights, pair_offsets, restore_shape):
        del topk_weights, pair_offsets
        output = torch.zeros(restore_shape, dtype=torch.bfloat16, device="cuda")
        for wave in waves:
            output.index_add_(0, wave.token_indices, wave.outputs)
        return output


def test_serial_wave_waits_before_reusing_compute_slots(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    for expert_id in range(runtime.num_experts):
        runtime.host_w13[expert_id].fill_(expert_id + 1)
    hidden = torch.zeros((2, 2), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64, device="cuda")
    weights = torch.ones_like(ids, dtype=torch.float32)

    output = execute_main_cache_waves(
        runtime,
        hidden,
        ids,
        weights,
        seam=_DelayedSlotReadSeam(),
        overlap=False,
    )

    expected = torch.tensor([[3, 3], [7, 7]], dtype=torch.bfloat16, device="cuda")
    torch.testing.assert_close(output, expected, rtol=0, atol=0)
