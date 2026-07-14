from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .errors import LayoutMismatchError
from .manifest import OffloadManifest, TensorLayout


@dataclass(frozen=True)
class BoundParameter:
    layer_id: int
    name: str
    original_device: torch.device


class PinnedHostStore:
    """One contiguous BF16 slab containing every manifest-selected tensor."""

    def __init__(self, manifest: OffloadManifest, *, pin_memory: bool = True):
        self.manifest = manifest
        self._slab = torch.empty(
            manifest.total_elements,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=pin_memory,
        )
        self._views: dict[tuple[int, str], torch.Tensor] = {}
        self._bindings: dict[tuple[int, str], BoundParameter] = {}
        for layer in manifest.layers:
            for layout in layer.tensors:
                self._views[(layer.layer_id, layout.name)] = self._make_view(layout)

    @property
    def slab(self) -> torch.Tensor:
        return self._slab

    @property
    def is_pinned(self) -> bool:
        return self._slab.is_pinned()

    @property
    def bindings(self) -> tuple[BoundParameter, ...]:
        return tuple(self._bindings.values())

    def _make_view(self, layout: TensorLayout) -> torch.Tensor:
        flat = self._slab.narrow(0, layout.offset_elements, layout.numel)
        return torch.as_strided(flat, size=layout.shape, stride=layout.stride)

    def tensor_view(self, layer_id: int, name: str) -> torch.Tensor:
        try:
            return self._views[(layer_id, name)]
        except KeyError as exc:
            raise KeyError(f"no host tensor for layer={layer_id}, name={name}") from exc

    def bind_parameter(
        self, layer_id: int, name: str, parameter: nn.Parameter
    ) -> BoundParameter:
        view = self.tensor_view(layer_id, name)
        if tuple(parameter.shape) != tuple(view.shape):
            raise LayoutMismatchError(
                f"shape mismatch for layer={layer_id}, parameter={name}: "
                f"expected={tuple(view.shape)}, actual={tuple(parameter.shape)}"
            )
        if parameter.dtype != view.dtype:
            raise LayoutMismatchError(
                f"dtype mismatch for layer={layer_id}, parameter={name}: "
                f"expected={view.dtype}, actual={parameter.dtype}"
            )
        if tuple(parameter.stride()) != tuple(view.stride()):
            raise LayoutMismatchError(
                f"stride mismatch for layer={layer_id}, parameter={name}: "
                f"expected={tuple(view.stride())}, "
                f"actual={tuple(parameter.stride())}"
            )
        key = (layer_id, name)
        if key in self._bindings:
            raise LayoutMismatchError(
                f"parameter already bound for layer={layer_id}, name={name}"
            )
        binding = BoundParameter(
            layer_id=layer_id,
            name=name,
            original_device=parameter.device,
        )
        parameter.data = view
        self._bindings[key] = binding
        return binding
