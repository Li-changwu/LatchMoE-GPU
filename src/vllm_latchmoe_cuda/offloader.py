from __future__ import annotations

import os
from collections.abc import Generator
from types import MethodType
from typing import Any

import torch
from torch import nn
from vllm.model_executor.offloader.base import BaseOffloader
from vllm.model_executor.offloader.uva import UVAOffloader

from .host_store import PinnedHostStore
from .capabilities import shared_expert_weight_bytes
from .manifest import ModelIdentityLock, OffloadManifest
from .profile import JsonlEventWriter
from .residency_plan import CudaResidencyPlan
from .runtime import CudaLayerRuntime


_EXPERT_WEIGHT_NAMES = frozenset({"w13_weight", "w2_weight"})
# Deprecated diagnostic constant retained for old benchmark readers.  It is not
# consulted by the production plan path.
TOTAL_OFFLOAD_BUDGET_BYTES = 14 * 1024**3
WAVE_SLOTS_ENV = "VLLM_LATCHMOE_WAVE_SLOTS"


def _wave_slot_count(main_slots: int) -> int:
    """Diagnostic/oracle slot override; production plans do not call this."""
    value = os.getenv(WAVE_SLOTS_ENV)
    wave_slots = min(main_slots, 32) if value is None else int(value)
    if wave_slots <= 0 or wave_slots > main_slots:
        raise ValueError(
            f"{WAVE_SLOTS_ENV} must be in [1, {main_slots}], got {wave_slots}"
        )
    return wave_slots
def _post_load_named_parameters(
    module: nn.Module,
    prefix: str = "",
    recurse: bool = True,
    remove_duplicate: bool = True,
):
    for name, parameter in nn.Module.named_parameters(
        module,
        prefix=prefix,
        recurse=recurse,
        remove_duplicate=remove_duplicate,
    ):
        if name.rsplit(".", 1)[-1] not in _EXPERT_WEIGHT_NAMES:
            yield name, parameter


def _install_post_load_filter(experts_module: nn.Module) -> None:
    if "named_parameters" in experts_module.__dict__:
        raise RuntimeError("module already has an instance named_parameters override")
    experts_module.named_parameters = MethodType(
        _post_load_named_parameters, experts_module
    )
    experts_module._latchmoe_post_load_filter = True


def _remove_post_load_filter(experts_module: nn.Module) -> None:
    if not getattr(experts_module, "_latchmoe_post_load_filter", False):
        raise RuntimeError("LatchMoE post-load filter is missing")
    delattr(experts_module, "named_parameters")
    delattr(experts_module, "_latchmoe_post_load_filter")


def _resolve_parameter(module: nn.Module, dotted_name: str) -> nn.Parameter:
    current: Any = module
    parts = dotted_name.split(".")
    for part in parts:
        current = getattr(current, part)
    if not isinstance(current, nn.Parameter):
        raise TypeError(f"{dotted_name} did not resolve to nn.Parameter")
    return current


def _move_unbound_state_to_device(
    module: nn.Module,
    *,
    device: torch.device,
    bound_parameter_ids: set[int],
) -> None:
    for parameter in module.parameters():
        if id(parameter) not in bound_parameter_ids and parameter.device != device:
            parameter.data = parameter.data.to(device)
    for buffer in module.buffers():
        if buffer.device != device:
            buffer.data = buffer.data.to(device)


class CudaSEWOffloader(BaseOffloader):
    """Plan-selected CPU-first offloader with one main cache per layer."""

    def __init__(
        self,
        manifest: OffloadManifest | None = None,
        *,
        plan: CudaResidencyPlan | None = None,
        identity_lock: ModelIdentityLock | None = None,
        pin_memory: bool = True,
        first_layer_id: int = 0,
        residual_uva_max_bytes: int = 0,
    ):
        if (manifest is None) == (plan is None):
            raise TypeError("provide exactly one of manifest or plan")
        self.manifest = manifest
        self.plan = plan
        self.identity_lock = identity_lock
        self.host_store = PinnedHostStore(
            plan if plan is not None else manifest, pin_memory=pin_memory
        )
        self.first_layer_id = first_layer_id
        if residual_uva_max_bytes < 0:
            raise ValueError("residual_uva_max_bytes must be non-negative")
        self.residual_uva = (
            UVAOffloader(cpu_offload_max_bytes=residual_uva_max_bytes)
            if residual_uva_max_bytes
            else None
        )
        self.bound_parameter_names: set[str] = set()
        self.bound_layers: dict[int, nn.Module] = {}
        self.runtimes: dict[int, CudaLayerRuntime] = {}
        self._wrapped = False
        self._post_initialized = False
        self._closed = False
        profile_path = os.getenv("VLLM_LATCHMOE_PROFILE_PATH")
        self.event_writer = JsonlEventWriter(profile_path) if profile_path else None

    @property
    def selected_layer_ids(self) -> tuple[int, ...]:
        if self.plan is not None:
            return self.plan.offloaded_layer_ids
        assert self.manifest is not None
        return self.manifest.layer_ids

    @property
    def num_experts(self) -> int:
        if self.plan is not None:
            return self.plan.num_experts
        assert self.manifest is not None
        return self.manifest.model.num_experts

    @property
    def num_slots(self) -> int:
        if self.plan is not None:
            return self.plan.effective_num_slots
        assert self.manifest is not None
        return self.manifest.num_slots

    def _parameter_layouts(self, layer_id: int) -> tuple[tuple[str, str], ...]:
        if self.plan is not None:
            return (
                ("w13_weight", "mlp.experts.w13_weight"),
                ("w2_weight", "mlp.experts.w2_weight"),
            )
        assert self.manifest is not None
        return tuple(
            (tensor.name, tensor.parameter_name)
            for tensor in self.manifest.layer(layer_id).tensors
        )

    def wrap_modules(
        self, modules_generator: Generator[nn.Module, None, None]
    ) -> list[nn.Module]:
        if self._wrapped:
            raise RuntimeError("wrap_modules may only be called once")
        self._wrapped = True
        modules: list[nn.Module] = []
        selected = set(self.selected_layer_ids)
        modules_iterator = iter(modules_generator)
        relative_index = 0
        while True:
            layer_id = self.first_layer_id + relative_index
            target_device = torch.get_default_device()
            construct_on_cpu = layer_id in selected and target_device.type != "cpu"
            try:
                if construct_on_cpu:
                    with torch.device("cpu"):
                        module = next(modules_iterator)
                else:
                    module = next(modules_iterator)
            except StopIteration:
                break
            modules.append(module)
            if layer_id not in selected:
                relative_index += 1
                continue
            bound_parameter_ids: set[int] = set()
            for tensor_name, parameter_name in self._parameter_layouts(layer_id):
                parameter = _resolve_parameter(module, parameter_name)
                self.host_store.bind_parameter(
                    layer_id,
                    tensor_name,
                    parameter,
                    original_device=target_device if construct_on_cpu else None,
                )
                bound_parameter_ids.add(id(parameter))
                self.bound_parameter_names.add(
                    f"model.layers.{layer_id}.{parameter_name}"
                )
            if construct_on_cpu:
                _move_unbound_state_to_device(
                    module,
                    device=target_device,
                    bound_parameter_ids=bound_parameter_ids,
                )
            _install_post_load_filter(module.get_submodule("mlp.experts"))
            self.bound_layers[layer_id] = module
            relative_index += 1
        if self.residual_uva is not None:
            filtered_modules = tuple(self.bound_layers.values())
            for module in filtered_modules:
                _install_post_load_filter(module)
            try:
                modules = self.residual_uva.wrap_modules(iter(modules))
            finally:
                for module in filtered_modules:
                    _remove_post_load_filter(module)
            if self.event_writer is not None:
                self.event_writer.write(
                    "residual_uva",
                    cpu_offload_max_bytes=self.residual_uva.cpu_offload_max_bytes,
                    cpu_offload_bytes=self.residual_uva.cpu_offload_bytes,
                )
        return modules

    def post_init(self) -> None:
        if self._post_initialized:
            raise RuntimeError("post_init may only be called once")
        missing = sorted(set(self.selected_layer_ids) - set(self.bound_layers))
        if missing:
            raise RuntimeError(f"missing manifest layers during binding: {missing}")
        self._post_initialized = True
        for layer_id, module in self.bound_layers.items():
            experts_module = module.get_submodule("mlp.experts")
            _remove_post_load_filter(experts_module)
            bindings = {
                binding.name: binding
                for binding in self.host_store.bindings
                if binding.layer_id == layer_id
            }
            device = bindings["w13_weight"].original_device
            host_w13 = self.host_store.tensor_view(layer_id, "w13_weight")
            host_w2 = self.host_store.tensor_view(layer_id, "w2_weight")
            if self.event_writer is not None:
                intermediate = host_w2.shape[-1]
                self.event_writer.write(
                    "host_store_sample",
                    layer_id=layer_id,
                    gate_0_0=float(host_w13[0, 0, 0]),
                    up_0_0=float(host_w13[0, intermediate, 0]),
                    down_0_0=float(host_w2[0, 0, 0]),
                    gate_last=float(host_w13[-1, intermediate - 1, -1]),
                    up_last=float(host_w13[-1, -1, -1]),
                    down_last=float(host_w2[-1, -1, -1]),
                )
            self.runtimes[layer_id] = CudaLayerRuntime(
                layer_id=layer_id,
                num_experts=self.num_experts,
                num_slots=self.num_slots,
                host_store=self.host_store,
                experts_module=experts_module,
                device=device,
                event_writer=self.event_writer,
            )
            self.runtimes[layer_id].plan_id = (
                None if self.plan is None else self.plan.plan_id
            )
            self.runtimes[layer_id].production_plan = self.plan is not None
            resident_shared_bytes = shared_expert_weight_bytes(
                getattr(experts_module, "_shared_experts", None)
            )
            host_routed_bytes = sum(binding.nbytes for binding in bindings.values())
            dynamic_slot_bytes = int(
                self.runtimes[layer_id].slot_w13.numel()
                * self.runtimes[layer_id].slot_w13.element_size()
                + self.runtimes[layer_id].slot_w2.numel()
                * self.runtimes[layer_id].slot_w2.element_size()
            )
            self.runtimes[layer_id].resident_shared_weight_bytes = resident_shared_bytes
            self.runtimes[layer_id].host_routed_expert_bytes = host_routed_bytes
            self.runtimes[layer_id].dynamic_slot_bytes = dynamic_slot_bytes
            if self.event_writer is not None:
                self.event_writer.write(
                    "residency_ledger",
                    layer_id=layer_id,
                    resident_shared_weight_bytes=resident_shared_bytes,
                    dynamic_slot_bytes=dynamic_slot_bytes,
                    host_routed_expert_bytes=host_routed_bytes,
                )
            if hasattr(experts_module, "router") and hasattr(
                experts_module, "quant_method"
            ):
                from .runner_adapter import install_vllm_forward_adapter

                install_vllm_forward_adapter(experts_module, self.runtimes[layer_id])

    def close(self) -> None:
        if self._closed:
            return
        close_error: BaseException | None = None
        try:
            for runtime in self.runtimes.values():
                try:
                    runtime.close()
                except BaseException as exc:
                    if close_error is None:
                        close_error = exc
            try:
                from .graph_ops import unregister_graph_runtime

                for runtime in self.runtimes.values():
                    unregister_graph_runtime(runtime)
            except BaseException as exc:
                if close_error is None:
                    close_error = exc
            if self.event_writer is not None:
                self.event_writer.close()
        finally:
            self._closed = True
        if close_error is not None:
            raise close_error
