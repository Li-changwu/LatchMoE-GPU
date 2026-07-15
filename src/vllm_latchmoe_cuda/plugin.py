from __future__ import annotations

import importlib
import os
from importlib import metadata
from pathlib import Path

from .errors import UnsupportedVllmVersionError
from .manifest import OffloadManifest
from .offloader import TOTAL_OFFLOAD_BUDGET_BYTES, CudaSEWOffloader
from .uva import ManifestUVAOffloader


SUPPORTED_VLLM_VERSION = "0.19.1"
MODE_ENV = "VLLM_LATCHMOE_MODE"
MANIFEST_ENV = "VLLM_LATCHMOE_MANIFEST"


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
    runner_module = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    current_factory = runner_module.create_offloader
    if getattr(current_factory, "_latchmoe_wrapped", False):
        return

    def create_offloader(offload_config):
        mode = os.getenv(MODE_ENV, "").strip().lower()
        if not mode:
            return current_factory(offload_config)
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
