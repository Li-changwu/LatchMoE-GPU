import hashlib
import json
from pathlib import Path

import pytest
import torch
from torch import nn

from vllm_latchmoe_cuda.manifest import OffloadManifest, canonical_json_bytes


@pytest.fixture
def tiny_manifest(tmp_path: Path) -> OffloadManifest:
    payload = {
        "schema_version": 1,
        "model": {
            "path": "/models/tiny",
            "revision": "tiny-revision",
            "config_sha256": "a" * 64,
            "weight_index_sha256": "b" * 64,
            "num_experts": 4,
        },
        "vllm": {
            "version": "0.19.1",
            "tag_commit": "b1388b1fbf5aaef47937fabe98931211684666a6",
        },
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "num_slots": 2,
        "layers": [
            {
                "layer_id": 0,
                "tensors": [
                    {
                        "name": "w13_weight",
                        "parameter_name": "mlp.experts.w13_weight",
                        "shape": [4, 4, 2],
                        "stride": [8, 2, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 0,
                        "numel": 32,
                        "nbytes": 64,
                    },
                    {
                        "name": "w2_weight",
                        "parameter_name": "mlp.experts.w2_weight",
                        "shape": [4, 2, 2],
                        "stride": [4, 2, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 32,
                        "numel": 16,
                        "nbytes": 32,
                    },
                ],
            }
        ],
    }
    document = dict(payload)
    document["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    path = tmp_path / "tiny_manifest.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    return OffloadManifest.load(path)


def direct_weight_loader(param: nn.Parameter, loaded: torch.Tensor, *args, **kwargs):
    param.data.copy_(loaded.to(device=param.device, dtype=param.dtype))
    return True


class TinyExperts(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.w13_weight = nn.Parameter(
            torch.empty((4, 4, 2), dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.w2_weight = nn.Parameter(
            torch.empty((4, 2, 2), dtype=torch.bfloat16, device=device),
            requires_grad=False,
        )
        self.w13_weight.weight_loader = direct_weight_loader
        self.w2_weight.weight_loader = direct_weight_loader


class TinyMlp(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.experts = TinyExperts(device)


class TinyDecoder(nn.Module):
    def __init__(self, device: str):
        super().__init__()
        self.mlp = TinyMlp(device)


@pytest.fixture
def tiny_decoder_factory():
    return TinyDecoder

