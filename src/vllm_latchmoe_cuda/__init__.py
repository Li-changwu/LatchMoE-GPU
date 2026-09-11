"""CUDA LatchMoE plugin for vLLM 0.19.1."""

from .manifest import OffloadManifest
from .residency_plan import CudaResidencyPlan, LayerExpertBytes

__all__ = ["CudaResidencyPlan", "LayerExpertBytes", "OffloadManifest"]
__version__ = "0.1.0"
