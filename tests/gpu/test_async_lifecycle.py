from dataclasses import replace

import pytest
import torch

from vllm_latchmoe_cuda.core.slots import SlotState
from vllm_latchmoe_cuda.errors import NoEvictableSlotError, StaleMappingError
from vllm_latchmoe_cuda.manifest import LayerLayout
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.transfer import ExpertCopy, contiguous_copy_runs


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


def test_contiguous_copy_runs_coalesce_adjacent_experts_and_slots():
    loads = (
        ExpertCopy(expert_id=4, slot_id=0, generation=1),
        ExpertCopy(expert_id=5, slot_id=1, generation=1),
        ExpertCopy(expert_id=8, slot_id=2, generation=1),
    )

    assert contiguous_copy_runs(loads) == (loads[:2], loads[2:])


def test_async_stage_uses_distinct_stream_and_marks_slots_ready(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)

    snapshot = runtime.stage_async((0, 1))

    assert (
        runtime.transfer_engine.stream.cuda_stream
        != torch.cuda.current_stream().cuda_stream
    )
    assert all(
        runtime.bank.slots[slot_id].state is SlotState.READY
        for slot_id in snapshot.slot_ids
    )
    runtime.validate_snapshot(snapshot)


def test_compute_done_event_guards_slot_reuse(tiny_manifest, tiny_decoder_factory):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    snapshot = runtime.stage_async((0, 1))
    compute = runtime.begin_compute(snapshot)
    pending = runtime.end_compute_async(compute)

    with pytest.raises(NoEvictableSlotError):
        runtime.stage_async((2, 3))

    runtime.wait_compute_done(pending)
    next_snapshot = runtime.stage_async((2, 3))

    assert next_snapshot.active_experts == (2, 3)


def test_async_map_publication_uses_pinned_lifetime_guard(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)

    first = runtime.stage_async((0, 1))

    assert runtime.pending_map_copy_count >= 1
    assert all(copy.cpu_map.is_pinned() for copy in runtime._pending_map_copies)
    runtime.stage_async((0, 1))
    with pytest.raises(StaleMappingError, match="mapping version"):
        runtime.validate_snapshot(first)


def test_shared_main_slots_wait_and_reload_across_layers(
    tiny_manifest, tiny_decoder_factory
):
    first_layout = tiny_manifest.layers[0]
    second_layout = LayerLayout(
        layer_id=1,
        tensors=tuple(
            replace(tensor, offset_elements=tensor.offset_elements + 48)
            for tensor in first_layout.tensors
        ),
    )
    manifest = replace(tiny_manifest, layers=(first_layout, second_layout))
    modules = (tiny_decoder_factory("cuda"), tiny_decoder_factory("cuda"))
    offloader = CudaSEWOffloader(manifest)
    offloader.wrap_modules(iter(modules))
    for name in ("w13_weight", "w2_weight"):
        offloader.host_store.tensor_view(0, name).zero_()
        offloader.host_store.tensor_view(1, name).fill_(1)
    offloader.post_init()
    first = offloader.runtimes[0]
    second = offloader.runtimes[1]

    first.prepare_compute_async((0, 1))
    first.finish_compute_async()
    second_snapshot = second.stage_async((0, 1))
    torch.cuda.synchronize()

    assert not first._pending_computes
    assert torch.count_nonzero(second.slot_w13[second_snapshot.slot_ids]).item() > 0

    first_snapshot = first.stage_async((0, 1))
    torch.cuda.synchronize()

    assert torch.count_nonzero(first.slot_w13[first_snapshot.slot_ids]).item() == 0
