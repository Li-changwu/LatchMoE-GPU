from __future__ import annotations

from collections.abc import Generator
from typing import Any

from torch import nn
from vllm.model_executor.offloader.base import BaseOffloader

from .host_store import PinnedHostStore
from .manifest import OffloadManifest
from .runtime import CudaLayerRuntime


def _resolve_parameter(module: nn.Module, dotted_name: str) -> nn.Parameter:
    current: Any = module
    parts = dotted_name.split(".")
    for part in parts:
        current = getattr(current, part)
    if not isinstance(current, nn.Parameter):
        raise TypeError(f"{dotted_name} did not resolve to nn.Parameter")
    return current


class CudaSEWOffloader(BaseOffloader):
    """Manifest-selected CUDA expert offloader attached by make_layers()."""

    def __init__(
        self,
        manifest: OffloadManifest,
        *,
        pin_memory: bool = True,
        first_layer_id: int = 0,
    ):
        self.manifest = manifest
        self.host_store = PinnedHostStore(manifest, pin_memory=pin_memory)
        self.first_layer_id = first_layer_id
        self.bound_parameter_names: set[str] = set()
        self.bound_layers: dict[int, nn.Module] = {}
        self.runtimes: dict[int, CudaLayerRuntime] = {}
        self._wrapped = False
        self._post_initialized = False

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
            layout = self.manifest.layer(layer_id)
            for tensor in layout.tensors:
                parameter = _resolve_parameter(module, tensor.parameter_name)
                self.host_store.bind_parameter(layer_id, tensor.name, parameter)
                self.bound_parameter_names.add(
                    f"model.layers.{layer_id}.{tensor.parameter_name}"
                )
            self.bound_layers[layer_id] = module
        return modules

    def post_init(self) -> None:
        if self._post_initialized:
            raise RuntimeError("post_init may only be called once")
        self._post_initialized = True
        for layer_id, module in self.bound_layers.items():
            experts_module = module.get_submodule("mlp.experts")
            bindings = {
                binding.name: binding
                for binding in self.host_store.bindings
                if binding.layer_id == layer_id
            }
            device = bindings["w13_weight"].original_device
            self.runtimes[layer_id] = CudaLayerRuntime(
                layer=self.manifest.layer(layer_id),
                num_experts=self.manifest.model.num_experts,
                num_slots=self.manifest.num_slots,
                host_store=self.host_store,
                experts_module=experts_module,
                device=device,
            )
