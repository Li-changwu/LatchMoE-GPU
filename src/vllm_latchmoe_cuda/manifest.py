from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import math
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Mapping, Sequence

from .errors import IdentityLockError, ManifestHashError, ManifestValidationError


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
    def load(
        cls, path: str | Path, *, diagnostic_mode: bool = True
    ) -> OffloadManifest:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(document.get("schema_version", 0)) == 1 and not diagnostic_mode:
            raise ManifestValidationError(
                "schema v1 manifests are diagnostic-only; use a v2 identity lock"
            )
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


@dataclass(frozen=True)
class ModelIdentityLock:
    """Content identity shared by the parent and every vLLM worker."""

    schema_version: int
    model_path: str
    revision: str
    config_sha256: str
    weight_index_sha256: str
    shard_files: tuple[tuple[str, int, str], ...]
    vllm_version: str
    vllm_source_sha256: str
    plan_id: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _vllm_source_hash() -> str:
    try:
        module = importlib.import_module("vllm.model_executor.layers.fused_moe")
        source = Path(inspect.getfile(module))
        if source.is_file():
            return _sha256_file(source)
    except (ImportError, OSError, TypeError):
        pass
    return "unavailable"


def _model_arg(vllm_args: Sequence[str]) -> str | None:
    args = list(vllm_args)
    for index, value in enumerate(args):
        if value == "--model" and index + 1 < len(args):
            return args[index + 1]
        if value.startswith("--model="):
            return value.split("=", 1)[1]
    return None


def serialize_identity_lock(lock: ModelIdentityLock) -> str:
    payload = {
        "schema_version": lock.schema_version,
        "model_path": lock.model_path,
        "revision": lock.revision,
        "config_sha256": lock.config_sha256,
        "weight_index_sha256": lock.weight_index_sha256,
        "shard_files": [list(item) for item in lock.shard_files],
        "vllm_version": lock.vllm_version,
        "vllm_source_sha256": lock.vllm_source_sha256,
        "plan_id": lock.plan_id,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def deserialize_identity_lock(raw: str) -> ModelIdentityLock:
    try:
        payload = json.loads(raw)
        lock = ModelIdentityLock(
            schema_version=int(payload["schema_version"]),
            model_path=str(payload["model_path"]),
            revision=str(payload["revision"]),
            config_sha256=str(payload["config_sha256"]),
            weight_index_sha256=str(payload["weight_index_sha256"]),
            shard_files=tuple(
                (str(item[0]), int(item[1]), str(item[2]))
                for item in payload["shard_files"]
            ),
            vllm_version=str(payload["vllm_version"]),
            vllm_source_sha256=str(payload["vllm_source_sha256"]),
            plan_id=str(payload["plan_id"]),
        )
    except (TypeError, ValueError, KeyError, json.JSONDecodeError) as exc:
        raise IdentityLockError("invalid identity lock JSON") from exc
    if lock.schema_version != 2:
        raise IdentityLockError(
            f"unsupported identity lock schema={lock.schema_version}"
        )
    if len(lock.plan_id) != 64:
        raise IdentityLockError("identity lock plan_id is malformed")
    return lock


def build_identity_lock(plan: Any, vllm_args: Sequence[str]) -> ModelIdentityLock:
    fingerprint = plan.model_fingerprint
    model_path_value = fingerprint.get("model_path") or fingerprint.get("path")
    model_path = str(model_path_value or _model_arg(vllm_args) or "")
    if not model_path:
        raise IdentityLockError("--model is required to build the identity lock")
    root = Path(model_path)
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    config_hash = str(fingerprint.get("config_sha256", ""))
    index_hash = str(fingerprint.get("weight_index_sha256", ""))
    if config.is_file():
        config_hash = _sha256_file(config)
    if index.is_file():
        index_hash = _sha256_file(index)
    if len(config_hash) != 64 or len(index_hash) != 64:
        raise IdentityLockError("model config and weight index must be hashable")
    shard_files: list[tuple[str, int, str]] = []
    if index.is_file():
        try:
            weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
            names = sorted({str(value) for value in weight_map.values()})
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise IdentityLockError("invalid model weight index") from exc
        for name in names:
            path = root / name
            if not path.is_file():
                raise IdentityLockError(f"model shard is missing: {path}")
            shard_files.append((name, path.stat().st_size, _sha256_file(path)))
    try:
        version = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        version = "unavailable"
    return ModelIdentityLock(
        schema_version=2,
        model_path=str(root),
        revision=str(fingerprint.get("revision", "unknown")),
        config_sha256=config_hash,
        weight_index_sha256=index_hash,
        shard_files=tuple(shard_files),
        vllm_version=version,
        vllm_source_sha256=_vllm_source_hash(),
        plan_id=str(plan.plan_id),
    )


def validate_identity_lock(lock: ModelIdentityLock, plan: Any) -> None:
    if lock.schema_version != 2:
        raise IdentityLockError("only identity lock schema v2 is supported")
    if lock.plan_id != plan.plan_id:
        raise IdentityLockError(
            f"identity lock plan mismatch: expected={plan.plan_id}, actual={lock.plan_id}"
        )
    root = Path(lock.model_path)
    config = root / "config.json"
    index = root / "model.safetensors.index.json"
    if not config.is_file() or _sha256_file(config) != lock.config_sha256:
        raise IdentityLockError("config.json SHA-256 mismatch")
    if not index.is_file() or _sha256_file(index) != lock.weight_index_sha256:
        raise IdentityLockError("model.safetensors.index.json SHA-256 mismatch")
    for name, expected_size, expected_hash in lock.shard_files:
        path = root / name
        if not path.is_file():
            raise IdentityLockError(f"model shard is missing: {path}")
        if path.stat().st_size != expected_size or _sha256_file(path) != expected_hash:
            raise IdentityLockError(f"model shard identity mismatch: {name}")
    try:
        actual_version = importlib_metadata.version("vllm")
    except importlib_metadata.PackageNotFoundError:
        actual_version = "unavailable"
    if lock.vllm_version != actual_version:
        raise IdentityLockError(
            f"vLLM version mismatch: expected={lock.vllm_version}, actual={actual_version}"
        )
    if lock.vllm_source_sha256 != _vllm_source_hash():
        raise IdentityLockError("vLLM seam source identity mismatch")
