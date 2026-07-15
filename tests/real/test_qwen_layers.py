import json
import os
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.manifest import OffloadManifest
from vllm_latchmoe_cuda.real_weights import compare_qwen_layer, layer_checkpoint_keys


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(
    os.environ.get(
        "LATCHMOE_REAL_MANIFEST",
        str(ROOT / "benchmark/manifests/offload_manifest.json"),
    )
)
LAYER_IDS = OffloadManifest.load(MANIFEST).layer_ids


def test_checkpoint_index_contains_every_manifest_expert_tensor():
    manifest = OffloadManifest.load(MANIFEST)
    index = json.loads(
        (Path(manifest.model.path) / "model.safetensors.index.json").read_text()
    )["weight_map"]

    for layer_id in manifest.layer_ids:
        keys = layer_checkpoint_keys(layer_id, num_experts=manifest.model.num_experts)
        assert len(keys) == 384
        assert set(keys).issubset(index)


@pytest.mark.real_model
@pytest.mark.cuda
@pytest.mark.skipif(
    os.getenv("LATCHMOE_RUN_REAL") != "1",
    reason="set LATCHMOE_RUN_REAL=1 to run real Qwen layer comparisons",
)
@pytest.mark.parametrize("layer_id", LAYER_IDS)
def test_real_qwen_layer_matches_staged_eager_and_waves(layer_id, tmp_path):
    result = compare_qwen_layer(
        manifest_path=MANIFEST,
        layer_id=layer_id,
        artifact_path=tmp_path / f"layer_{layer_id}.json",
    )

    assert result["eager_close"] is True
    assert result["waves_close"] is True
