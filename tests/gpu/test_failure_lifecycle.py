import json

import pytest
import torch

from vllm_latchmoe_cuda.errors import RuntimePoisonedError
from vllm_latchmoe_cuda.moe_seam import FunctionalMoeSeam
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.profile import JsonlEventWriter
from vllm_latchmoe_cuda.runtime import RuntimeState
from vllm_latchmoe_cuda.runner_adapter import execute_main_cache_waves


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


class FailingAfterEnqueueSeam(FunctionalMoeSeam):
    def __init__(self):
        super().__init__(combine_fn=lambda **kwargs: kwargs["waves"][0].outputs)
        self.calls = 0

    def run_expert_mlp(self, **kwargs):
        self.calls += 1
        payload = super().run_expert_mlp(**kwargs)
        if self.calls == 3:
            raise RuntimeError("synthetic kernel callback failure")
        return payload


def _offloader(tiny_manifest, tiny_decoder_factory, profile_path):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.event_writer = JsonlEventWriter(profile_path)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        offloader.host_store.tensor_view(0, name).normal_()
    offloader.post_init()
    runtime = offloader.runtimes[0]
    runtime.plan_id = "test-plan"
    return offloader, runtime


def test_kernel_failure_poison_rejects_next_forward_and_drains(
    tiny_manifest, tiny_decoder_factory, tmp_path
):
    offloader, runtime = _offloader(
        tiny_manifest, tiny_decoder_factory, tmp_path / "failure.jsonl"
    )
    hidden = torch.randn((1, 2), dtype=torch.bfloat16, device="cuda")
    weights = torch.tensor([[0.5, 0.5]], device="cuda")
    first_ids = torch.tensor([[0, 1]], dtype=torch.int64, device="cuda")
    second_ids = torch.tensor([[0, 2]], dtype=torch.int64, device="cuda")
    seam = FailingAfterEnqueueSeam()

    execute_main_cache_waves(runtime, hidden, first_ids, weights, seam=seam)
    with pytest.raises(RuntimeError, match="synthetic kernel callback failure"):
        execute_main_cache_waves(runtime, hidden, second_ids, weights, seam=seam)

    assert runtime.state is RuntimeState.POISONED
    mapping = runtime.log2phy.detach().cpu().clone()
    with pytest.raises(RuntimePoisonedError):
        execute_main_cache_waves(runtime, hidden, first_ids, weights, seam=seam)
    assert torch.equal(runtime.log2phy.detach().cpu(), mapping)

    offloader.close()
    offloader.close()
    assert runtime.state is RuntimeState.CLOSED
    events = [
        json.loads(line)
        for line in (tmp_path / "failure.jsonl").read_text().splitlines()
    ]
    failures = [event for event in events if event["event"] == "failure"]
    assert failures
    assert failures[-1]["exception_type"] == "RuntimeError"
    assert failures[-1]["plan_id"] == "test-plan"
    assert failures[-1]["layer_id"] == 0
    assert failures[-1]["active_experts"] == [2]
    assert failures[-1]["leases"]
    assert failures[-1]["drained"] is True
