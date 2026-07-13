import pytest
import torch

from vllm_latchmoe_cuda.errors import ActiveExpertCapacityError, StaleMappingError
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _runtime(tiny_manifest, tiny_decoder_factory):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for parameter in module.mlp.experts.parameters():
        parameter.data.copy_(
            torch.arange(parameter.numel(), dtype=torch.bfloat16).view_as(parameter)
        )
    offloader.post_init()
    return offloader.runtimes[0]


def test_sync_stage_copies_exact_expert_and_updates_map(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)

    snapshot = runtime.stage_sync((3, 1))

    mapping = runtime.log2phy.cpu().tolist()
    assert mapping[3] >= 0
    assert mapping[1] >= 0
    assert mapping[0] == -1
    assert torch.equal(
        runtime.slot_w13[mapping[3]], runtime.host_w13[3].to(device="cuda")
    )
    runtime.validate_snapshot(snapshot)


def test_stage_rejects_union_larger_than_slots(tiny_manifest, tiny_decoder_factory):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)

    with pytest.raises(ActiveExpertCapacityError, match="active_count=3"):
        runtime.stage_sync((0, 1, 2))


def test_stale_snapshot_is_detected_after_slot_reuse(
    tiny_manifest, tiny_decoder_factory
):
    runtime = _runtime(tiny_manifest, tiny_decoder_factory)
    stale = runtime.stage_sync((0, 1))
    runtime.stage_sync((2, 3))

    with pytest.raises(StaleMappingError):
        runtime.validate_snapshot(stale)

