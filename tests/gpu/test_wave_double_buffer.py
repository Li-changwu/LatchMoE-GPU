import pytest
import torch
from dataclasses import replace

from vllm_latchmoe_cuda.manifest import LayerLayout
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import execute_exact_waves


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_wave_executor_alternates_two_stable_stage_banks(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    pointers = runtime.stage_pool.data_ptrs()
    hidden = torch.randn((4, 2), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device="cuda")
    weights = torch.full((4, 2), 0.5, device="cuda")

    execute_exact_waves(runtime, hidden, ids, weights)
    first_trace = runtime.last_wave_trace
    execute_exact_waves(runtime, hidden, ids, weights)

    assert first_trace.buffer_by_wave == ((0, 0), (1, 1))
    assert runtime.stage_pool.data_ptrs() == pointers
    assert (
        runtime.stage_pool.banks[0].w13.data_ptr()
        != runtime.stage_pool.banks[1].w13.data_ptr()
    )


def test_transfer_aware_prefetch_never_changes_compute_order(
    tiny_manifest, tiny_decoder_factory
):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    runtime = offloader.runtimes[0]
    hidden = torch.randn((4, 2), dtype=torch.bfloat16, device="cuda")
    ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device="cuda")
    weights = torch.full((4, 2), 0.5, device="cuda")

    execute_exact_waves(runtime, hidden, ids, weights, transfer_aware=True)

    assert runtime.last_wave_trace.compute_order == (0, 1)
    assert sorted(runtime.last_wave_trace.issue_order) == [0, 1]


def test_two_layers_share_the_same_double_stage_pool(
    tiny_manifest, tiny_decoder_factory
):
    first = tiny_manifest.layers[0]
    second = LayerLayout(
        layer_id=1,
        tensors=tuple(
            replace(tensor, offset_elements=tensor.offset_elements + 48)
            for tensor in first.tensors
        ),
    )
    manifest = replace(tiny_manifest, layers=(first, second))
    modules = (tiny_decoder_factory("cuda"), tiny_decoder_factory("cuda"))
    offloader = CudaSEWOffloader(manifest)
    offloader.wrap_modules(iter(modules))
    for layer_id in manifest.layer_ids:
        for name in ("w13_weight", "w2_weight"):
            host = offloader.host_store.tensor_view(layer_id, name)
            host.copy_(torch.randn_like(host))

    offloader.post_init()

    assert offloader.runtimes[0].stage_pool is offloader.runtimes[1].stage_pool
    assert offloader.runtimes[0].stage_pool is offloader.stage_pool
    assert len(offloader.stage_pool.banks) == 2
