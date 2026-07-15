from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .errors import ManifestHashError, ManifestValidationError


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def document_with_hash(payload: Mapping[str, Any]) -> dict[str, Any]:
    document = dict(payload)
    document["manifest_sha256"] = hashlib.sha256(
        canonical_json_bytes(payload)
    ).hexdigest()
    return document


@dataclass(frozen=True)
class ModelIdentity:
    path: str
    revision: str
    config_sha256: str
    weight_index_sha256: str
    num_experts: int


@dataclass(frozen=True)
class VllmIdentity:
    version: str
    tag_commit: str


@dataclass(frozen=True)
class TensorLayout:
    name: str
    parameter_name: str
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: str
    offset_elements: int
    numel: int
    nbytes: int

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> TensorLayout:
        return cls(
            name=str(payload["name"]),
            parameter_name=str(payload["parameter_name"]),
            shape=tuple(int(value) for value in payload["shape"]),
            stride=tuple(int(value) for value in payload["stride"]),
            dtype=str(payload["dtype"]),
            offset_elements=int(payload["offset_elements"]),
            numel=int(payload["numel"]),
            nbytes=int(payload["nbytes"]),
        )


@dataclass(frozen=True)
class LayerLayout:
    layer_id: int
    tensors: tuple[TensorLayout, ...]

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> LayerLayout:
        return cls(
            layer_id=int(payload["layer_id"]),
            tensors=tuple(
                TensorLayout.from_payload(value) for value in payload["tensors"]
            ),
        )

    def tensor(self, name: str) -> TensorLayout:
        for tensor in self.tensors:
            if tensor.name == name:
                return tensor
        raise KeyError(f"layer {self.layer_id} has no tensor {name!r}")


@dataclass(frozen=True)
class OffloadManifest:
    schema_version: int
    manifest_sha256: str
    model: ModelIdentity
    vllm: VllmIdentity
    dtype: str
    tensor_parallel_size: int
    num_slots: int
    layers: tuple[LayerLayout, ...]

    @classmethod
    def load(cls, path: str | Path) -> OffloadManifest:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            expected = str(document.pop("manifest_sha256"))
        except KeyError as exc:
            raise ManifestValidationError("manifest_sha256 is required") from exc
        actual = hashlib.sha256(canonical_json_bytes(document)).hexdigest()
        if actual != expected:
            raise ManifestHashError(expected=expected, actual=actual)
        manifest = cls.from_payload(document, manifest_sha256=expected)
        manifest._validate()
        return manifest

    @classmethod
    def from_payload(
        cls, payload: Mapping[str, Any], *, manifest_sha256: str
    ) -> OffloadManifest:
        model = payload["model"]
        vllm = payload["vllm"]
        return cls(
            schema_version=int(payload["schema_version"]),
            manifest_sha256=manifest_sha256,
            model=ModelIdentity(
                path=str(model["path"]),
                revision=str(model["revision"]),
                config_sha256=str(model["config_sha256"]),
                weight_index_sha256=str(model["weight_index_sha256"]),
                num_experts=int(model["num_experts"]),
            ),
            vllm=VllmIdentity(
                version=str(vllm["version"]), tag_commit=str(vllm["tag_commit"])
            ),
            dtype=str(payload["dtype"]),
            tensor_parallel_size=int(payload["tensor_parallel_size"]),
            num_slots=int(payload["num_slots"]),
            layers=tuple(
                LayerLayout.from_payload(value) for value in payload["layers"]
            ),
        )

    @property
    def layer_ids(self) -> tuple[int, ...]:
        return tuple(layer.layer_id for layer in self.layers)

    @property
    def total_elements(self) -> int:
        return max(
            (
                tensor.offset_elements + tensor.numel
                for layer in self.layers
                for tensor in layer.tensors
            ),
            default=0,
        )

    @property
    def parameter_names(self) -> frozenset[str]:
        return frozenset(
            f"model.layers.{layer.layer_id}.{tensor.parameter_name}"
            for layer in self.layers
            for tensor in layer.tensors
        )

    def layer(self, layer_id: int) -> LayerLayout:
        for layer in self.layers:
            if layer.layer_id == layer_id:
                return layer
        raise KeyError(f"layer {layer_id} is not in the offload manifest")

    def validate_runtime(
        self, *, vllm_version: str, dtype: str, tensor_parallel_size: int
    ) -> None:
        if vllm_version != self.vllm.version:
            raise ManifestValidationError(
                f"vLLM version mismatch: expected={self.vllm.version}, "
                f"actual={vllm_version}"
            )
        if dtype != self.dtype:
            raise ManifestValidationError(
                f"dtype mismatch: expected={self.dtype}, actual={dtype}"
            )
        if tensor_parallel_size != self.tensor_parallel_size:
            raise ManifestValidationError(
                "tensor parallel size mismatch: "
                f"expected={self.tensor_parallel_size}, actual={tensor_parallel_size}"
            )

    def validate_model_files(self) -> None:
        model_path = Path(self.model.path)
        checks = (
            ("config.json", self.model.config_sha256),
            ("model.safetensors.index.json", self.model.weight_index_sha256),
        )
        for filename, expected in checks:
            path = model_path / filename
            if not path.is_file():
                raise ManifestValidationError(f"model file is missing: {path}")
            digest = hashlib.sha256()
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if actual != expected:
                raise ManifestValidationError(
                    f"{filename} SHA-256 mismatch: expected={expected}, actual={actual}"
                )

    def _validate(self) -> None:
        if self.schema_version != 1:
            raise ManifestValidationError(
                f"unsupported schema_version={self.schema_version}"
            )
        if self.dtype != "bfloat16":
            raise ManifestValidationError("only bfloat16 manifests are supported")
        if self.tensor_parallel_size != 1:
            raise ManifestValidationError("only tensor_parallel_size=1 is supported")
        if self.num_slots <= 0:
            raise ManifestValidationError("num_slots must be positive")
        if self.num_slots > self.model.num_experts:
            raise ManifestValidationError("num_slots cannot exceed model.num_experts")
        if not self.layers:
            raise ManifestValidationError("at least one offload layer is required")
        if self.layer_ids != tuple(sorted(set(self.layer_ids))):
            raise ManifestValidationError("layer ids must be unique and sorted")

        ranges: list[tuple[int, int, str]] = []
        for layer in self.layers:
            names: set[str] = set()
            for tensor in layer.tensors:
                label = f"layer={layer.layer_id}, tensor={tensor.name}"
                if tensor.name in names:
                    raise ManifestValidationError(f"duplicate tensor name: {label}")
                names.add(tensor.name)
                if not tensor.shape or len(tensor.shape) != len(tensor.stride):
                    raise ManifestValidationError(f"invalid shape/stride: {label}")
                if tensor.dtype != self.dtype:
                    raise ManifestValidationError(f"dtype mismatch: {label}")
                if tensor.numel != math.prod(tensor.shape):
                    raise ManifestValidationError(f"numel mismatch: {label}")
                if tensor.nbytes != tensor.numel * 2:
                    raise ManifestValidationError(f"nbytes mismatch: {label}")
                if tensor.offset_elements < 0:
                    raise ManifestValidationError(f"negative offset: {label}")
                ranges.append(
                    (
                        tensor.offset_elements,
                        tensor.offset_elements + tensor.numel,
                        label,
                    )
                )
            if names != {"w13_weight", "w2_weight"}:
                raise ManifestValidationError(
                    f"layer {layer.layer_id} must contain w13_weight and w2_weight"
                )

        ranges.sort()
        for previous, current in zip(ranges, ranges[1:]):
            if current[0] < previous[1]:
                raise ManifestValidationError(
                    f"host ranges overlap: {previous[2]} and {current[2]}"
                )
