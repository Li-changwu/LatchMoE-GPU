import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm_latchmoe_cuda.manifest import (
    OffloadManifest,
    canonical_json_bytes,
)
from vllm_latchmoe_cuda.offloader import CudaSEWOffloader
from vllm_latchmoe_cuda.runner_adapter import (
    _capturable_weights_moe,
    execute_exact_waves,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


def _expert_mlp(x, w13, w2):
    gate_up = torch.matmul(w13.float(), x.float())
    gate, up = gate_up.chunk(2, dim=0)
    return torch.matmul(w2.float(), F.silu(gate) * up)


def test_two_exact_waves_match_full_reference(tiny_manifest, tiny_decoder_factory):
    torch.manual_seed(31)
    module = tiny_decoder_factory("cuda")
    offloader = CudaSEWOffloader(tiny_manifest)
    offloader.wrap_modules(iter((module,)))
    host_w13 = offloader.host_store.tensor_view(0, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(0, "w2_weight")
    host_w13.copy_(torch.randn_like(host_w13))
    host_w2.copy_(torch.randn_like(host_w2))
    full_w13 = host_w13.clone().to("cuda")
    full_w2 = host_w2.clone().to("cuda")
    offloader.post_init()
    runtime = offloader.runtimes[0]
    hidden = torch.randn((4, 2), dtype=torch.bfloat16, device="cuda")
    topk_ids = torch.tensor([[0, 1], [2, 3], [0, 2], [1, 3]], device="cuda")
    topk_weights = torch.tensor(
        [[0.7, 0.3], [0.4, 0.6], [0.2, 0.8], [0.55, 0.45]],
        dtype=torch.float32,
        device="cuda",
    )
    expected = torch.zeros((4, 2), dtype=torch.float32, device="cuda")
    for token in range(hidden.shape[0]):
        for position in range(topk_ids.shape[1]):
            expert = int(topk_ids[token, position])
            expected[token] += topk_weights[token, position] * _expert_mlp(
                hidden[token], full_w13[expert], full_w2[expert]
            )

    actual = execute_exact_waves(runtime, hidden, topk_ids, topk_weights)

    assert actual.dtype is torch.bfloat16
    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    assert runtime.last_wave_trace.pair_count == topk_ids.numel()
    assert runtime.last_wave_trace.compute_order == (0, 1)


class _ManyExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty((128, 4, 2), dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty((128, 2, 2), dtype=torch.bfloat16, device="cuda"),
            requires_grad=False,
        )


class _ManyMlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = _ManyExperts()


class _ManyDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = _ManyMlp()


def _many_manifest(tmp_path):
    import hashlib
    import json

    payload = {
        "schema_version": 1,
        "model": {
            "path": "/models/tiny-128",
            "revision": "tiny-128",
            "config_sha256": "a" * 64,
            "weight_index_sha256": "b" * 64,
            "num_experts": 128,
        },
        "vllm": {
            "version": "0.19.1",
            "tag_commit": "b1388b1fbf5aaef47937fabe98931211684666a6",
        },
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "num_slots": 32,
        "layers": [
            {
                "layer_id": 0,
                "tensors": [
                    {
                        "name": "w13_weight",
                        "parameter_name": "mlp.experts.w13_weight",
                        "shape": [128, 4, 2],
                        "stride": [8, 2, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 0,
                        "numel": 1024,
                        "nbytes": 2048,
                    },
                    {
                        "name": "w2_weight",
                        "parameter_name": "mlp.experts.w2_weight",
                        "shape": [128, 2, 2],
                        "stride": [4, 2, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 1024,
                        "numel": 512,
                        "nbytes": 1024,
                    },
                ],
            }
        ],
    }
    document = dict(payload)
    document["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    path = tmp_path / "many_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return OffloadManifest.load(path)


@pytest.mark.parametrize("workload", ["prefill", "mixed"])
def test_union128_gpu_waves_match_full_reference(tmp_path, workload):
    torch.manual_seed(37)
    manifest = _many_manifest(tmp_path)
    module = _ManyDecoder()
    offloader = CudaSEWOffloader(manifest)
    offloader.wrap_modules(iter((module,)))
    host_w13 = offloader.host_store.tensor_view(0, "w13_weight")
    host_w2 = offloader.host_store.tensor_view(0, "w2_weight")
    host_w13.copy_(torch.randn_like(host_w13))
    host_w2.copy_(torch.randn_like(host_w2))
    full_w13 = host_w13.clone().to("cuda")
    full_w2 = host_w2.clone().to("cuda")
    offloader.post_init()
    runtime = offloader.runtimes[0]
    prefill_ids = torch.arange(128, device="cuda").view(16, 8)
    if workload == "mixed":
        decode_ids = torch.arange(8, device="cuda").view(1, 8).repeat(4, 1)
        topk_ids = torch.cat((decode_ids, prefill_ids), dim=0)
    else:
        topk_ids = prefill_ids
    topk_weights = torch.rand(topk_ids.shape, device="cuda")
    topk_weights /= topk_weights.sum(dim=-1, keepdim=True)
    hidden = torch.randn(
        (topk_ids.shape[0], 2), dtype=torch.bfloat16, device="cuda"
    )
    expected = _capturable_weights_moe(
        full_w13, full_w2, hidden, topk_ids, topk_weights
    )

    actual = execute_exact_waves(runtime, hidden, topk_ids, topk_weights)

    torch.testing.assert_close(actual.float(), expected, rtol=2e-2, atol=2e-2)
    assert runtime.last_wave_trace.pair_count == topk_ids.numel()
    assert runtime.last_wave_trace.compute_order == (0, 1, 2, 3)
