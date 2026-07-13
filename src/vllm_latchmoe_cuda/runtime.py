from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .core.expert_key import ExpertKey
from .core.policy import LruPolicy
from .core.slots import ComputeHandle, ExpertSlotBank, SlotLease, SlotState
from .errors import (
    ActiveExpertCapacityError,
    StableAddressError,
    StaleMappingError,
)
from .host_store import PinnedHostStore
from .manifest import LayerLayout
from .profile import RuntimeCounters
from .transfer import copy_expert_sync


@dataclass(frozen=True)
class LayerMappingSnapshot:
    layer_id: int
    active_experts: tuple[int, ...]
    leases: tuple[SlotLease, ...]
    mapping_version: int

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(lease.slot_id for lease in self.leases)


class CudaLayerRuntime:
    def __init__(
        self,
        *,
        layer: LayerLayout,
        num_experts: int,
        num_slots: int,
        host_store: PinnedHostStore,
        experts_module: nn.Module,
        device: torch.device,
    ):
        if device.type != "cuda":
            raise ValueError(f"CUDA runtime requires a CUDA device, got {device}")
        self.layer_id = layer.layer_id
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.host_w13 = host_store.tensor_view(layer.layer_id, "w13_weight")
        self.host_w2 = host_store.tensor_view(layer.layer_id, "w2_weight")
        w13_parameter = getattr(experts_module, "w13_weight")
        w2_parameter = getattr(experts_module, "w2_weight")
        slot_w13 = torch.empty(
            (num_slots, *self.host_w13.shape[1:]),
            dtype=self.host_w13.dtype,
            device=device,
        )
        slot_w2 = torch.empty(
            (num_slots, *self.host_w2.shape[1:]),
            dtype=self.host_w2.dtype,
            device=device,
        )
        w13_parameter.data = slot_w13
        w2_parameter.data = slot_w2
        self.slot_w13_parameter = w13_parameter
        self.slot_w2_parameter = w2_parameter
        self.slot_w13 = w13_parameter
        self.slot_w2 = w2_parameter
        self.log2phy = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=device
        )
        self.expert_map = self.log2phy
        if "_expert_map" in experts_module._buffers:
            experts_module._buffers["_expert_map"] = self.expert_map
        else:
            experts_module.register_buffer("_expert_map", self.expert_map)
        if hasattr(experts_module, "local_num_experts"):
            experts_module.local_num_experts = num_slots
        if hasattr(experts_module, "n_local_physical_experts"):
            experts_module.n_local_physical_experts = num_slots
        self.bank = ExpertSlotBank(num_slots)
        self.policy = LruPolicy()
        self.counters = RuntimeCounters()
        self.mapping_version = 0
        self._stable_ptrs = self.data_ptrs()

    def data_ptrs(self) -> dict[str, int]:
        return {
            "slot_w13": self.slot_w13.data_ptr(),
            "slot_w2": self.slot_w2.data_ptr(),
            "log2phy": self.log2phy.data_ptr(),
            "expert_map": self.expert_map.data_ptr(),
        }

    def assert_stable_addresses(self) -> None:
        actual = self.data_ptrs()
        if actual != self._stable_ptrs:
            raise StableAddressError(
                f"layer {self.layer_id} tensor address changed: "
                f"expected={self._stable_ptrs}, actual={actual}"
            )

    def _normalize_active(self, active_experts: Iterable[int]) -> tuple[int, ...]:
        active = tuple(dict.fromkeys(int(value) for value in active_experts))
        invalid = tuple(
            value for value in active if value < 0 or value >= self.num_experts
        )
        if invalid:
            raise ValueError(
                f"layer {self.layer_id} has invalid expert ids: {invalid}"
            )
        if len(active) > self.num_slots:
            raise ActiveExpertCapacityError(
                f"layer={self.layer_id}, active_count={len(active)}, "
                f"slot_count={self.num_slots}"
            )
        return active

    def stage_sync(self, active_experts: Iterable[int]) -> LayerMappingSnapshot:
        self.assert_stable_addresses()
        active = self._normalize_active(active_experts)
        selected: dict[int, SlotLease] = {}
        reserved: set[int] = set()
        for expert_id in active:
            key = ExpertKey(self.layer_id, expert_id)
            slot = self.bank.find(key)
            if slot is None:
                continue
            if slot.state is SlotState.COMPUTING:
                raise StaleMappingError(
                    f"layer={self.layer_id}, expert={expert_id} is still computing"
                )
            self.bank.touch(slot.slot_id)
            selected[expert_id] = SlotLease(slot.slot_id, key, slot.generation)
            reserved.add(slot.slot_id)
            self.counters.increment("slot_hit")

        for expert_id in active:
            if expert_id in selected:
                continue
            slot_id = self.policy.choose(self.bank, excluded=reserved)
            slot = self.bank.slots[slot_id]
            if slot.state is SlotState.READY:
                self.bank.evict(slot_id)
                self.counters.increment("eviction")
            key = ExpertKey(self.layer_id, expert_id)
            lease = self.bank.begin_load(key, slot_id=slot_id)
            copy_expert_sync(
                host_w13=self.host_w13,
                host_w2=self.host_w2,
                expert_id=expert_id,
                slot_w13=self.slot_w13,
                slot_w2=self.slot_w2,
                slot_id=slot_id,
            )
            lease = self.bank.mark_ready(slot_id, lease.generation)
            selected[expert_id] = lease
            reserved.add(slot_id)
            self.counters.increment("slot_miss")
            self.counters.increment(
                "h2d_bytes",
                self.host_w13[expert_id].numel()
                * self.host_w13.element_size()
                + self.host_w2[expert_id].numel() * self.host_w2.element_size(),
            )

        cpu_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        for key, slot_id in self.bank.ready_mapping().items():
            if key.layer_id == self.layer_id:
                cpu_map[key.expert_id] = slot_id
        self.log2phy.copy_(cpu_map, non_blocking=False)
        self.mapping_version += 1
        snapshot = LayerMappingSnapshot(
            layer_id=self.layer_id,
            active_experts=active,
            leases=tuple(selected[expert] for expert in active),
            mapping_version=self.mapping_version,
        )
        self.validate_snapshot(snapshot)
        self.assert_stable_addresses()
        return snapshot

    def validate_snapshot(self, snapshot: LayerMappingSnapshot) -> None:
        if snapshot.layer_id != self.layer_id:
            raise StaleMappingError(
                f"snapshot layer mismatch: expected={self.layer_id}, "
                f"actual={snapshot.layer_id}"
            )
        for expert_id, lease in zip(snapshot.active_experts, snapshot.leases):
            self.bank.validate_generation(lease.slot_id, lease.generation)
            mapped = int(self.log2phy[expert_id].item())
            if mapped != lease.slot_id:
                raise StaleMappingError(
                    f"stale expert mapping: layer={self.layer_id}, "
                    f"expert={expert_id}, expected_slot={lease.slot_id}, "
                    f"actual_slot={mapped}"
                )

    def begin_compute(self, snapshot: LayerMappingSnapshot) -> ComputeHandle:
        self.validate_snapshot(snapshot)
        return self.bank.begin_compute(snapshot.slot_ids)

    def end_compute(self, handle: ComputeHandle) -> None:
        self.bank.end_compute(handle)

