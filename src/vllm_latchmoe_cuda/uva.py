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
        if self.use_uva and not self.pin_memory:
            raise RuntimeError("UVA requires pinned CPU memory")
        self.first_layer_id = first_layer_id
        self.offloaded_parameter_names: set[str] = set()
        self._cpu_tensors: dict[tuple[int, str], torch.Tensor] = {}
        self._wrapped = False

    @property
    def cpu_backing_tensors(self) -> tuple[torch.Tensor, ...]:
        return tuple(self._cpu_tensors.values())

    def cpu_tensor(self, layer_id: int, name: str) -> torch.Tensor:
        return self._cpu_tensors[(layer_id, name)]

    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]:
        if self._wrapped:
            raise RuntimeError("wrap_modules may only be called once")
        self._wrapped = True
        modules: list[nn.Module] = []
        selected = set(self.manifest.layer_ids)
        for relative_index, module in enumerate(modules_generator):
            layer_id = self.first_layer_id + relative_index
            modules.append(module)
            if layer_id not in selected:
                continue
            for layout in self.manifest.layer(layer_id).tensors:
                parameter = _resolve_parameter(module, layout.parameter_name)
                cpu_data = torch.empty_strided(
                    size=parameter.data.size(),
                    stride=parameter.data.stride(),
                    dtype=parameter.data.dtype,
                    layout=parameter.data.layout,
                    device="cpu",
                    pin_memory=self.pin_memory,
                )
                cpu_data.copy_(parameter.data)
                if self.use_uva:
                    parameter.data = get_accelerator_view_from_cpu_tensor(cpu_data)
                    parameter._vllm_is_uva_offloaded = True
                else:
                    parameter.data = cpu_data
                self._cpu_tensors[(layer_id, layout.name)] = cpu_data
                self.offloaded_parameter_names.add(
                    f"model.layers.{layer_id}.{layout.parameter_name}"
                )
        return modules

