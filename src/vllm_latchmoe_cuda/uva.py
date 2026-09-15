from __future__ import annotations

from collections.abc import Generator

import torch
from torch import nn
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.utils.platform_utils import is_pin_memory_available, is_uva_available
from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

from .manifest import OffloadManifest
from .offloader import _resolve_parameter


class ManifestUVAOffloader(BaseOffloader):
    """Native vLLM UVA data path with exact manifest parameter selection."""

    def __init__(
        self,
        manifest: OffloadManifest,
        *,
        pin_memory: bool | None = None,
        use_uva: bool | None = None,
        first_layer_id: int = 0,
    ):
        self.manifest = manifest
        self.pin_memory = (
            is_pin_memory_available() if pin_memory is None else bool(pin_memory)
        )
        self.use_uva = is_uva_available() if use_uva is None else bool(use_uva)
        if not self.pin_memory or not self.use_uva:
            raise RuntimeError("controlled UVA baseline requires pinned UVA")
        self.first_layer_id = first_layer_id
        self.offloaded_parameter_names: set[str] = set()
        self.cpu_offload_bytes = 0
        self.cpu_offload_max_bytes = sum(
            tensor.nbytes
            for layer in manifest.layers
            for tensor in layer.tensors
        )
        self._wrapped = False

    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]:
        if self._wrapped:
            raise RuntimeError("wrap_modules may only be called once")
        self._wrapped = True
        modules: list[nn.Module] = []
        selected = set(self.manifest.layer_ids)
        bound_layers: set[int] = set()
        for relative_index, module in enumerate(modules_generator):
            layer_id = self.first_layer_id + relative_index
            modules.append(module)
            if layer_id not in selected:
                continue
            bound_layers.add(layer_id)
            for layout in self.manifest.layer(layer_id).tensors:
                parameter = _resolve_parameter(module, layout.parameter_name)
                if tuple(parameter.shape) != layout.shape:
                    raise RuntimeError(
                        f"UVA shape mismatch: layer={layer_id}, "
                        f"tensor={layout.name}, expected={layout.shape}, "
                        f"actual={tuple(parameter.shape)}"
                    )
                if str(parameter.dtype).removeprefix("torch.") != layout.dtype:
                    raise RuntimeError(
                        f"UVA dtype mismatch: layer={layer_id}, "
                        f"tensor={layout.name}, expected={layout.dtype}, "
                        f"actual={parameter.dtype}"
                    )
                if tuple(parameter.stride()) != layout.stride:
                    raise RuntimeError(
                        f"UVA stride mismatch: layer={layer_id}, "
                        f"tensor={layout.name}, expected={layout.stride}, "
                        f"actual={tuple(parameter.stride())}"
                    )
                cpu_data = torch.empty_strided(
                    size=parameter.data.size(),
                    stride=parameter.data.stride(),
                    dtype=parameter.data.dtype,
                    layout=parameter.data.layout,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                parameter.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                parameter._vllm_is_uva_offloaded = True
                self.cpu_offload_bytes += parameter.numel() * parameter.element_size()
                self.offloaded_parameter_names.add(
                    f"model.layers.{layer_id}.{layout.parameter_name}"
                )
        missing = sorted(selected - bound_layers)
        if missing:
            raise RuntimeError(f"missing manifest layers during binding: {missing}")
        return modules
