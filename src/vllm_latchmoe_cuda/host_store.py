from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .errors import LayoutMismatchError
from .manifest import OffloadManifest, TensorLayout
from .residency_plan import CudaResidencyPlan


@dataclass(frozen=True)
class BoundParameter:
    layer_id: int
    name: str
    original_device: torch.device
    shape: tuple[int, ...] = ()
    stride: tuple[int, ...] = ()
    dtype: str = ""
    nbytes: int = 0


class PinnedHostStore:
    """Pinned selected-layer weights.

    Manifest input retains the old contiguous diagnostic layout.  Plan input
    uses independent views allocated from each parameter's real shape/stride;
    no global slab or unselected-layer storage is created.
    """

    def __init__(
        self,
        source: OffloadManifest | CudaResidencyPlan | None = None,
        *,
        pin_memory: bool = True,
    ):
        self.manifest = source if isinstance(source, OffloadManifest) else None
        self.plan = source if isinstance(source, CudaResidencyPlan) else None
        self._dynamic = self.plan is not None or source is None
        self._pin_memory = bool(pin_memory)
        self._slab = None
        if self.manifest is not None:
            try:
                self._slab = torch.empty(
                    self.manifest.total_elements,
                    dtype=torch.bfloat16,
                    device="cpu",
                    pin_memory=pin_memory,
                )
            except RuntimeError:
                self._slab = torch.empty(
                    self.manifest.total_elements, dtype=torch.bfloat16, device="cpu"
                )
        self._views: dict[tuple[int, str], torch.Tensor] = {}
        self._bindings: dict[tuple[int, str], BoundParameter] = {}
        if self.manifest is not None:
            for layer in self.manifest.layers:
                for layout in layer.tensors:
                    self._views[(layer.layer_id, layout.name)] = self._make_view(layout)

    @property
    def slab(self) -> torch.Tensor | None:
        return self._slab

    @property
    def is_pinned(self) -> bool:
        return bool(self._slab is not None and self._slab.is_pinned())

    @property
    def bindings(self) -> tuple[BoundParameter, ...]:
        return tuple(self._bindings.values())

    def _make_view(self, layout: TensorLayout) -> torch.Tensor:
        assert self._slab is not None
        flat = self._slab.narrow(0, layout.offset_elements, layout.numel)
        return torch.as_strided(flat, size=layout.shape, stride=layout.stride)

    def tensor_view(self, layer_id: int, name: str) -> torch.Tensor:
        try:
            return self._views[(layer_id, name)]
        except KeyError as exc:
            raise KeyError(f"no host tensor for layer={layer_id}, name={name}") from exc

    def bind_parameter(
        self,
        layer_id: int,
        name: str,
        parameter: nn.Parameter,
        *,
        original_device: torch.device | None = None,
    ) -> BoundParameter:
        key = (layer_id, name)
        if self._dynamic:
            if key in self._bindings:
                raise LayoutMismatchError(
                    f"parameter already bound for layer={layer_id}, name={name}"
                )
            shape = tuple(int(value) for value in parameter.shape)
            stride = tuple(int(value) for value in parameter.stride())
            try:
                view = torch.empty_strided(
                    shape,
                    stride,
                    dtype=parameter.dtype,
                    device="cpu",
                    pin_memory=self._pin_memory,
                )
            except RuntimeError:
                view = torch.empty_strided(
                    shape, stride, dtype=parameter.dtype, device="cpu"
                )
            self._views[key] = view
            parameter_device = parameter.device if original_device is None else original_device
            binding = BoundParameter(
                layer_id=layer_id,
                name=name,
                original_device=parameter_device,
                shape=shape,
                stride=stride,
                dtype=str(parameter.dtype).removeprefix("torch."),
                nbytes=parameter.numel() * parameter.element_size(),
            )
            parameter.data = view
            self._bindings[key] = binding
            self._validate_dynamic_layer(layer_id)
            return binding
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
            original_device=(
                parameter.device if original_device is None else original_device
            ),
            shape=tuple(parameter.shape),
            stride=tuple(parameter.stride()),
            dtype=str(parameter.dtype).removeprefix("torch."),
            nbytes=parameter.numel() * parameter.element_size(),
        )
        parameter.data = view
        self._bindings[key] = binding
        return binding

    def _validate_dynamic_layer(self, layer_id: int) -> None:
        if self.plan is None:
            return
        expected = next(
            (
                item.routed_expert_bytes
                for item in self.plan.layer_expert_bytes
                if item.layer_id == layer_id
            ),
            None,
        )
        if expected is None:
            raise LayoutMismatchError(f"layer {layer_id} is not in the residency plan")
        actual = sum(
            binding.nbytes
            for binding in self._bindings.values()
            if binding.layer_id == layer_id
        )
        # Check once both routed expert tensors have been bound.  A plan's
        # routed bytes describe w13 + w2, while shared expert bytes are absent.
        names = {
            binding.name
            for binding in self._bindings.values()
            if binding.layer_id == layer_id
        }
        if {"w13_weight", "w2_weight"}.issubset(names) and actual != expected:
            raise LayoutMismatchError(
                f"layer={layer_id} bytes mismatch: expected={expected}, actual={actual}"
            )
