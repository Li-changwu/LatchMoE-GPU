import hashlib
import json
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.errors import ManifestHashError, ManifestValidationError
from vllm_latchmoe_cuda.manifest import OffloadManifest, canonical_json_bytes


def _payload() -> dict[str, object]:
    return {
        "schema_version": 1,
        "model": {
            "path": "/models/qwen",
            "revision": "model-revision",
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
                "layer_id": 3,
                "tensors": [
                    {
                        "name": "w13_weight",
                        "parameter_name": "mlp.experts.w13_weight",
                        "shape": [4, 6, 2],
                        "stride": [12, 2, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 0,
                        "numel": 48,
                        "nbytes": 96,
                    },
                    {
                        "name": "w2_weight",
                        "parameter_name": "mlp.experts.w2_weight",
                        "shape": [4, 2, 3],
                        "stride": [6, 3, 1],
                        "dtype": "bfloat16",
                        "offset_elements": 48,
                        "numel": 24,
                        "nbytes": 48,
                    },
                ],
            }
        ],
    }


def _write_manifest(path: Path, payload: dict[str, object]) -> None:
    document = dict(payload)
    document["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    path.write_text(json.dumps(document), encoding="utf-8")


def test_manifest_loads_verified_layout(tmp_path: Path):
    path = tmp_path / "offload_manifest.json"
    _write_manifest(path, _payload())

    manifest = OffloadManifest.load(path)

    assert manifest.layer_ids == (3,)
    assert manifest.num_slots == 2
    assert manifest.total_elements == 72
    assert manifest.layers[0].tensor("w2_weight").shape == (4, 2, 3)


def test_manifest_hash_rejects_mutation(tmp_path: Path):
    path = tmp_path / "offload_manifest.json"
    _write_manifest(path, _payload())
    document = json.loads(path.read_text(encoding="utf-8"))
    document["num_slots"] = 3
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ManifestHashError, match="manifest SHA-256 mismatch"):
        OffloadManifest.load(path)


def test_manifest_rejects_overlapping_host_ranges(tmp_path: Path):
    payload = _payload()
    payload["layers"][0]["tensors"][1]["offset_elements"] = 47
    path = tmp_path / "offload_manifest.json"
    _write_manifest(path, payload)

    with pytest.raises(ManifestValidationError, match="overlap"):
        OffloadManifest.load(path)


def test_runtime_validation_rejects_wrong_vllm(tmp_path: Path):
    path = tmp_path / "offload_manifest.json"
    _write_manifest(path, _payload())
    manifest = OffloadManifest.load(path)

    with pytest.raises(ManifestValidationError, match="vLLM version"):
        manifest.validate_runtime(
            vllm_version="0.19.0", dtype="bfloat16", tensor_parallel_size=1
        )


def test_model_file_hash_validation_rejects_local_mutation(tmp_path: Path):
    model_path = tmp_path / "model"
    model_path.mkdir()
    config_path = model_path / "config.json"
    index_path = model_path / "model.safetensors.index.json"
    config_path.write_bytes(b'{"model_type":"qwen3_moe"}\n')
    index_path.write_bytes(b'{"weight_map":{}}\n')
    payload = _payload()
    payload["model"]["path"] = str(model_path)
    payload["model"]["config_sha256"] = hashlib.sha256(
        config_path.read_bytes()
    ).hexdigest()
    payload["model"]["weight_index_sha256"] = hashlib.sha256(
        index_path.read_bytes()
    ).hexdigest()
    manifest_path = tmp_path / "offload_manifest.json"
    _write_manifest(manifest_path, payload)
    manifest = OffloadManifest.load(manifest_path)

    manifest.validate_model_files()
    config_path.write_bytes(b'{"model_type":"changed"}\n')

    with pytest.raises(ManifestValidationError, match="config.json SHA-256"):
        manifest.validate_model_files()
