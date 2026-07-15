#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from vllm_latchmoe_cuda.manifest import document_with_hash


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL = Path("/root/models/Qwen3-30B-A3B-Instruct-2507")
DEFAULT_OUTPUT = ROOT / "benchmark/manifests/offload_manifest.json"
DEFAULT_REVISION = "0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe"
LAYERS = (3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_payload(
    model_path: Path,
    revision: str = DEFAULT_REVISION,
    layers: tuple[int, ...] = LAYERS,
) -> dict[str, object]:
    w13_shape = (128, 1536, 2048)
    w2_shape = (128, 2048, 768)
    w13_numel = 128 * 1536 * 2048
    w2_numel = 128 * 2048 * 768
    offset = 0
    layer_documents: list[dict[str, object]] = []
    for layer_id in layers:
        tensors = [
            {
                "name": "w13_weight",
                "parameter_name": "mlp.experts.w13_weight",
                "shape": list(w13_shape),
                "stride": [1536 * 2048, 2048, 1],
                "dtype": "bfloat16",
                "offset_elements": offset,
                "numel": w13_numel,
                "nbytes": w13_numel * 2,
            },
            {
                "name": "w2_weight",
                "parameter_name": "mlp.experts.w2_weight",
                "shape": list(w2_shape),
                "stride": [2048 * 768, 768, 1],
                "dtype": "bfloat16",
                "offset_elements": offset + w13_numel,
                "numel": w2_numel,
                "nbytes": w2_numel * 2,
            },
        ]
        layer_documents.append({"layer_id": layer_id, "tensors": tensors})
        offset += w13_numel + w2_numel

    return {
        "schema_version": 1,
        "model": {
            "path": str(model_path),
            "revision": revision,
            "config_sha256": sha256_file(model_path / "config.json"),
            "weight_index_sha256": sha256_file(
                model_path / "model.safetensors.index.json"
            ),
            "num_experts": 128,
        },
        "vllm": {
            "version": "0.19.1",
            "tag_commit": "b1388b1fbf5aaef47937fabe98931211684666a6",
        },
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "num_slots": 32,
        "layers": layer_documents,
    }


def render(
    model_path: Path,
    revision: str = DEFAULT_REVISION,
    layers: tuple[int, ...] = LAYERS,
) -> str:
    return (
        json.dumps(
            document_with_hash(build_payload(model_path, revision, layers)),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument(
        "--layers",
        default=",".join(str(layer_id) for layer_id in LAYERS),
        help="comma-separated sorted layer ids",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    layers = tuple(int(value) for value in args.layers.split(",") if value)
    if not layers or layers != tuple(sorted(set(layers))):
        parser.error("--layers must contain unique sorted layer ids")
    expected = render(args.model, args.revision, layers)
    if args.check:
        if not args.output.is_file() or args.output.read_text() != expected:
            print(f"manifest is stale: {args.output}")
            return 1
        print(f"manifest is current: {args.output}")
        return 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(expected, encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
