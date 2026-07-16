from __future__ import annotations

import importlib
import os
from types import MethodType
from importlib import metadata
from pathlib import Path

from .errors import UnsupportedVllmVersionError
from .manifest import OffloadManifest
from .offloader import TOTAL_OFFLOAD_BUDGET_BYTES, CudaSEWOffloader
from .profile import JsonlEventWriter
from .uva import ManifestUVAOffloader


SUPPORTED_VLLM_VERSION = "0.19.1"
MODE_ENV = "VLLM_LATCHMOE_MODE"
MANIFEST_ENV = "VLLM_LATCHMOE_MANIFEST"
TELEMETRY_ENV = "VLLM_LATCHMOE_TELEMETRY_PATH"
PROFILE_ENV = "VLLM_LATCHMOE_PROFILE_PATH"


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
        manifest = load_manifest_from_env()
        if mode == "latchmoe":
            residual_uva_max_bytes = max(
                0, TOTAL_OFFLOAD_BUDGET_BYTES - manifest.total_elements * 2
            )
            return CudaSEWOffloader(
                manifest, residual_uva_max_bytes=residual_uva_max_bytes
            )
        if mode == "uva":
            return ManifestUVAOffloader(manifest)
        raise ValueError(
            f"unsupported {MODE_ENV}={mode!r}; expected 'latchmoe' or 'uva'"
        )

    create_offloader._latchmoe_wrapped = True
    create_offloader._latchmoe_native_factory = current_factory
    runner_module.create_offloader = create_offloader
