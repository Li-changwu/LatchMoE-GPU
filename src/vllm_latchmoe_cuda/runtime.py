from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
from torch import nn

from .core.expert_key import ExpertKey
from .core.policy import LruPolicy
from .core.slots import ComputeHandle, ExpertSlotBank, SlotLease, SlotState
from .core.waves import WaveDescriptor
from .errors import (
    ActiveExpertCapacityError,
    NoEvictableSlotError,
    StableAddressError,
    StagingDuringCaptureError,
    StaleMappingError,
)
from .host_store import PinnedHostStore
from .manifest import LayerLayout
from .profile import RuntimeCounters
from .transfer import (
    CudaTransferEngine,
    ExpertCopy,
    TransferTicket,
    copy_expert_sync,
)


@dataclass(frozen=True)
class LayerMappingSnapshot:
    layer_id: int
    active_experts: tuple[int, ...]
    leases: tuple[SlotLease, ...]
    mapping_version: int

    @property
    def slot_ids(self) -> tuple[int, ...]:
        return tuple(lease.slot_id for lease in self.leases)


@dataclass(frozen=True)
class PendingCompute:
    handle: ComputeHandle
    event: torch.cuda.Event


@dataclass
class StageBank:
    bank_id: int
    w13: torch.Tensor
    w2: torch.Tensor
    compute_done: torch.cuda.Event | None = None


@dataclass(frozen=True)
class StagedWave:
    wave_id: int
    bank_id: int
    experts: tuple[int, ...]
    ticket: TransferTicket


@dataclass(frozen=True)
class WaveExecutionTrace:
    pair_count: int
    compute_order: tuple[int, ...]
    issue_order: tuple[int, ...]
    buffer_by_wave: tuple[tuple[int, int], ...]


class CudaStagePool:
    def __init__(
        self,
        *,
        device: torch.device,
        num_slots: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
        buffer_count: int = 2,
    ):
        if buffer_count != 2:
            raise ValueError("LatchMoE B2 requires exactly two stage banks")
        self.device = device
        self.num_slots = num_slots
        self.transfer_engine = CudaTransferEngine(device)
        self.banks = tuple(
            StageBank(
                bank_id=index,
                w13=torch.empty((num_slots, *w13_shape), dtype=dtype, device=device),
                w2=torch.empty((num_slots, *w2_shape), dtype=dtype, device=device),
            )
            for index in range(buffer_count)
        )

    def data_ptrs(self) -> tuple[tuple[int, int], ...]:
        return tuple((bank.w13.data_ptr(), bank.w2.data_ptr()) for bank in self.banks)

    def issue(
        self, runtime: CudaLayerRuntime, wave: WaveDescriptor, bank_id: int
    ) -> StagedWave:
        bank = self.banks[bank_id]
        if bank.compute_done is not None:
            self.transfer_engine.stream.wait_event(bank.compute_done)
        copies = tuple(
            ExpertCopy(expert_id=expert, slot_id=position, generation=0)
            for position, expert in enumerate(wave.experts)
        )
        ticket = self.transfer_engine.load_many_async(
            host_w13=runtime.host_w13,
            host_w2=runtime.host_w2,
            slot_w13=bank.w13,
            slot_w2=bank.w2,
            copies=copies,
        )
        return StagedWave(wave.wave_id, bank_id, wave.experts, ticket)

    def wait_ready(self, staged: StagedWave) -> StageBank:
        self.transfer_engine.wait_ready(staged.ticket)
        return self.banks[staged.bank_id]

    def record_compute_done(self, bank_id: int) -> None:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.device))
        self.banks[bank_id].compute_done = event


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
        stage_pool: CudaStagePool | None = None,
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
        if not hasattr(type(experts_module), "expert_map"):
            experts_module.expert_map = self.expert_map
        if hasattr(experts_module, "local_num_experts"):
            experts_module.local_num_experts = num_slots
        if hasattr(experts_module, "n_local_physical_experts"):
            experts_module.n_local_physical_experts = num_slots
        self.bank = ExpertSlotBank(num_slots)
        self.policy = LruPolicy()
        self.counters = RuntimeCounters()
        self.transfer_engine = CudaTransferEngine(device)
        self.stage_pool = stage_pool or CudaStagePool(
            device=device,
            num_slots=num_slots,
            w13_shape=tuple(self.host_w13.shape[1:]),
            w2_shape=tuple(self.host_w2.shape[1:]),
            dtype=self.host_w13.dtype,
        )
        self.last_wave_trace: WaveExecutionTrace | None = None
        self.mapping_version = 0
        self._active_compute: ComputeHandle | None = None
        self._pending_computes: list[PendingCompute] = []
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

        snapshot = self._publish_mapping(active, selected)
        self.validate_snapshot(snapshot)
        self.assert_stable_addresses()
        return snapshot

    def stage_async(self, active_experts: Iterable[int]) -> LayerMappingSnapshot:
        if torch.cuda.is_current_stream_capturing():
            raise StagingDuringCaptureError(
                f"dynamic staging attempted during CUDA Graph capture: "
                f"layer={self.layer_id}"
            )
        self.assert_stable_addresses()
        active = self._normalize_active(active_experts)
        selected: dict[int, SlotLease] = {}
        reserved: set[int] = set()
        copies: list[ExpertCopy] = []
        for expert_id in active:
            key = ExpertKey(self.layer_id, expert_id)
            slot = self.bank.find(key)
            if slot is None:
                continue
            if slot.state is SlotState.COMPUTING:
                raise NoEvictableSlotError(
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
            copies.append(ExpertCopy(expert_id, slot_id, lease.generation))
            selected[expert_id] = lease
            reserved.add(slot_id)
            self.counters.increment("slot_miss")

        if copies:
            ticket = self.transfer_engine.load_many_async(
                host_w13=self.host_w13,
                host_w2=self.host_w2,
                slot_w13=self.slot_w13,
                slot_w2=self.slot_w2,
                copies=copies,
            )
            self.transfer_engine.wait_ready(ticket)
            for copy in copies:
                lease = self.bank.mark_ready(copy.slot_id, copy.generation)
                selected[copy.expert_id] = lease
                self.counters.increment(
                    "h2d_bytes",
                    self.host_w13[copy.expert_id].numel()
                    * self.host_w13.element_size()
                    + self.host_w2[copy.expert_id].numel()
                    * self.host_w2.element_size(),
                )

        snapshot = self._publish_mapping(active, selected)
        self.validate_snapshot(snapshot)
        self.assert_stable_addresses()
        return snapshot

    def _publish_mapping(
        self, active: tuple[int, ...], selected: dict[int, SlotLease]
    ) -> LayerMappingSnapshot:
        cpu_map = torch.full((self.num_experts,), -1, dtype=torch.int32)
        for key, slot_id in self.bank.ready_mapping().items():
            if key.layer_id == self.layer_id:
                cpu_map[key.expert_id] = slot_id
        self.log2phy.copy_(cpu_map, non_blocking=False)
        self.mapping_version += 1
        return LayerMappingSnapshot(
            layer_id=self.layer_id,
            active_experts=active,
            leases=tuple(selected[expert] for expert in active),
            mapping_version=self.mapping_version,
        )

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

    def end_compute_async(self, handle: ComputeHandle) -> PendingCompute:
        event = torch.cuda.Event()
        event.record(torch.cuda.current_stream(self.slot_w13.device))
        return PendingCompute(handle=handle, event=event)

    def wait_compute_done(self, pending: PendingCompute) -> None:
        pending.event.synchronize()
        self.bank.end_compute(pending.handle)

    def prepare_compute_async(self, active_experts: Iterable[int]) -> None:
        if self._active_compute is not None:
            raise RuntimeError(f"layer {self.layer_id} already has active compute")
        self.release_pending_for_transfer()
        snapshot = self.stage_async(active_experts)
        self._active_compute = self.begin_compute(snapshot)

    def finish_compute_async(self) -> PendingCompute:
        if self._active_compute is None:
            raise RuntimeError(f"layer {self.layer_id} has no active compute")
        pending = self.end_compute_async(self._active_compute)
        self._active_compute = None
        self._pending_computes.append(pending)
        return pending

    def release_pending_for_transfer(self) -> None:
        if not self._pending_computes:
            return
        for pending in self._pending_computes:
            self.transfer_engine.stream.wait_event(pending.event)
            self.bank.end_compute(pending.handle)
        self._pending_computes.clear()
