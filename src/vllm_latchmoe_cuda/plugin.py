from __future__ import annotations

import importlib
import atexit
import os
from types import MethodType
from importlib import metadata
from pathlib import Path

from .errors import UnsupportedVllmVersionError
from .manifest import (
    OffloadManifest,
    deserialize_identity_lock,
    validate_identity_lock,
)
from .offloader import CudaSEWOffloader
from .residency_plan import deserialize_residency_plan
from .profile import JsonlEventWriter
from .uva import ManifestUVAOffloader


SUPPORTED_VLLM_VERSION = "0.19.1"
MODE_ENV = "VLLM_LATCHMOE_MODE"
MANIFEST_ENV = "VLLM_LATCHMOE_MANIFEST"
RESIDENCY_PLAN_ENV = "VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"
IDENTITY_LOCK_ENV = "VLLM_LATCHMOE_IDENTITY_LOCK_JSON"
TELEMETRY_ENV = "VLLM_LATCHMOE_TELEMETRY_PATH"
PROFILE_ENV = "VLLM_LATCHMOE_PROFILE_PATH"
UVA_RESERVATION_ENV = "VLLM_LATCHMOE_UVA_RESERVATION_BYTES"


def _instrument_cudagraph_evidence() -> None:
    evidence_path = os.getenv(PROFILE_ENV) or os.getenv(TELEMETRY_ENV)
    if not evidence_path:
        return
    from vllm.compilation.cuda_graph import CUDAGraphWrapper
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    if getattr(CUDAGraphWrapper, "_latchmoe_evidence_wrapped", False):
        return
    original_call = CUDAGraphWrapper.__call__
    writer = JsonlEventWriter(evidence_path)
    evidence: dict[str, set[str]] = {"capture": set(), "replay": set()}

    def call(self, *args, **kwargs):
        wrapper_mode = self.runtime_mode.name
        if (
            wrapper_mode in evidence["capture"]
            and wrapper_mode in evidence["replay"]
        ):
            return original_call(self, *args, **kwargs)
        descriptor = None
        runtime_mode = None
        had_graph = False
        if is_forward_context_available():
            context = get_forward_context()
            descriptor = context.batch_descriptor
            runtime_mode = context.cudagraph_runtime_mode
            if descriptor is not None and runtime_mode == self.runtime_mode:
                entry = self.concrete_cudagraph_entries.get(descriptor)
                had_graph = entry is not None and entry.cudagraph is not None
        output = original_call(self, *args, **kwargs)
        if descriptor is None or runtime_mode != self.runtime_mode:
            return output
        if had_graph and wrapper_mode not in evidence["replay"]:
            writer.write(
                "cudagraph_replay",
                runtime_mode=self.runtime_mode.name,
                batch_descriptor=str(descriptor),
            )
            evidence["replay"].add(wrapper_mode)
        elif wrapper_mode not in evidence["capture"]:
            entry = self.concrete_cudagraph_entries.get(descriptor)
            if entry is not None and entry.cudagraph is not None:
                writer.write(
                    "cudagraph_capture",
                    runtime_mode=self.runtime_mode.name,
                    batch_descriptor=str(descriptor),
                )
                evidence["capture"].add(wrapper_mode)
        return output

    CUDAGraphWrapper.__call__ = call
    CUDAGraphWrapper._latchmoe_evidence_wrapped = True


def _instrument_stock_uva(offloader):
    telemetry_path = os.getenv(TELEMETRY_ENV)
    if not telemetry_path:
        return offloader
    from vllm.model_executor.offloader.uva import UVAOffloader

    if not isinstance(offloader, UVAOffloader):
        return offloader
    original_wrap = offloader.wrap_modules
    writer = JsonlEventWriter(telemetry_path)

    def wrap_modules(self, modules_generator):
        modules = original_wrap(modules_generator)
        writer.write(
            "stock_uva",
            implementation=f"{type(self).__module__}.{type(self).__name__}",
            cpu_offload_max_bytes=self.cpu_offload_max_bytes,
            cpu_offload_bytes=self.cpu_offload_bytes,
        )
        return modules

    offloader.wrap_modules = MethodType(wrap_modules, offloader)
    return offloader


def _instrument_manifest_uva(offloader):
    telemetry_path = os.getenv(TELEMETRY_ENV)
    if not telemetry_path:
        return offloader
    writer = JsonlEventWriter(telemetry_path)
    original_wrap = offloader.wrap_modules

    def wrap_modules(self, modules_generator):
        modules = original_wrap(modules_generator)
        writer.write(
            "stock_uva",
            implementation=f"{type(self).__module__}.{type(self).__name__}",
            cpu_offload_max_bytes=self.cpu_offload_max_bytes,
            cpu_offload_bytes=self.cpu_offload_bytes,
            selection="manifest_exact",
            uva_reservation_bytes=self.reserved_hbm_bytes,
        )
        writer.close()
        return modules

    offloader.wrap_modules = MethodType(wrap_modules, offloader)
    return offloader


def load_manifest_from_env() -> OffloadManifest:
    value = os.getenv(MANIFEST_ENV)
    if not value:
        raise RuntimeError(f"{MANIFEST_ENV} must name the frozen offload manifest")
    manifest = OffloadManifest.load(Path(value))
    manifest.validate_model_files()
    return manifest


def register() -> None:
    actual_version = metadata.version("vllm")
    if actual_version != SUPPORTED_VLLM_VERSION:
        raise UnsupportedVllmVersionError(
            expected=SUPPORTED_VLLM_VERSION, actual=actual_version
        )
    _instrument_cudagraph_evidence()
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    current_factory = runner_module.create_offloader
    if getattr(current_factory, "_latchmoe_wrapped", False):
        return

    def create_offloader(offload_config):
        mode = os.getenv(MODE_ENV, "").strip().lower()
        if not mode:
            return _instrument_stock_uva(current_factory(offload_config))
        if mode == "latchmoe":
            graph_mode = os.getenv("VLLM_LATCHMOE_GRAPH_MODE", "piecewise").strip().lower()
            if graph_mode in {"full", "full_decode_only", "full_and_piecewise"}:
                raise RuntimeError(
                    "LatchMoE residency requires PIECEWISE graph boundaries; "
                    f"got {graph_mode!r}"
                )
            raw_plan = os.getenv(RESIDENCY_PLAN_ENV)
            if not raw_plan:
                raise RuntimeError(
                    "LatchMoE worker is missing the parent residency plan"
                )
            raw_lock = os.getenv(IDENTITY_LOCK_ENV)
            if not raw_lock:
                raise RuntimeError(
                    "LatchMoE worker is missing the parent identity lock"
                )
            plan = deserialize_residency_plan(raw_plan)
            identity_lock = deserialize_identity_lock(raw_lock)
            validate_identity_lock(identity_lock, plan)
            offloader = CudaSEWOffloader(plan=plan, identity_lock=identity_lock)
            atexit.register(offloader.close)
            return offloader
        if mode == "uva":
            manifest = load_manifest_from_env()
            raw_plan = os.getenv(RESIDENCY_PLAN_ENV)
            raw_lock = os.getenv(IDENTITY_LOCK_ENV)
            if raw_plan and raw_lock:
                plan = deserialize_residency_plan(raw_plan)
                identity_lock = deserialize_identity_lock(raw_lock)
                validate_identity_lock(identity_lock, plan)
                if tuple(manifest.layer_ids) != tuple(plan.offloaded_layer_ids):
                    raise RuntimeError("UVA manifest and parent plan selected layers differ")
            raw_reservation = os.getenv(UVA_RESERVATION_ENV, "0")
            try:
                reservation_bytes = int(raw_reservation)
            except ValueError as exc:
                raise ValueError(
                    f"{UVA_RESERVATION_ENV} must be an integer"
                ) from exc
            if reservation_bytes < 0:
                raise ValueError(
                    f"{UVA_RESERVATION_ENV} must be non-negative"
                )
            return _instrument_manifest_uva(
                ManifestUVAOffloader(
                    manifest, reservation_bytes=reservation_bytes
                )
            )
        raise ValueError(
            f"unsupported {MODE_ENV}={mode!r}; expected 'latchmoe' or 'uva'"
        )

    create_offloader._latchmoe_wrapped = True
    create_offloader._latchmoe_native_factory = current_factory
    runner_module.create_offloader = create_offloader
