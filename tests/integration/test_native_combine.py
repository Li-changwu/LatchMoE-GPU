import pytest
import torch

from vllm_latchmoe_cuda.moe_seam import NativeWavePayload, SpyMoeSeam
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import execute_main_cache_waves


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def test_three_waves_run_experts_and_combine_once(tiny_manifest, tiny_decoder_factory):
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    for name in ("w13_weight", "w2_weight"):
        host = offloader.host_store.tensor_view(0, name)
        host.copy_(torch.randn_like(host))
    offloader.post_init()
    runtime = offloader.runtimes[0]

    def native_combine(*, waves, topk_weights, pair_offsets, restore_shape):
        output = torch.zeros(restore_shape, device=pair_offsets.device, dtype=torch.float32)
        flat_weights = topk_weights.reshape(-1).float()
        for wave in waves:
            values = wave.outputs.float() * flat_weights.index_select(0, wave.pair_offsets).unsqueeze(-1)
            output.scatter_add_(0, wave.token_indices[:, None].expand_as(values), values)
        return output.to(dtype=torch.bfloat16)

    seam = SpyMoeSeam(native_combine)
    hidden = torch.randn((3, 2), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 2]], dtype=torch.int64, device="cuda")
    topk_weights = torch.full((3, 2), 0.5, dtype=torch.float32, device="cuda")

    output = execute_main_cache_waves(runtime, hidden, topk_ids, topk_weights, seam=seam)

    assert output.shape == hidden.shape
    assert seam.run_calls == 2
    assert seam.combine_calls == 1
    assert runtime.last_wave_trace.pair_count == topk_ids.numel()

