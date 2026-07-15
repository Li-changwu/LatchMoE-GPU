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
from .profile import JsonlEventWriter, RuntimeCounters
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


@dataclass(frozen=True)
class PendingMapCopy:
    cpu_map: torch.Tensor
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
        if torch.cuda.is_current_stream_capturing():
            raise StagingDuringCaptureError(
                f"wave staging attempted during CUDA Graph capture: "
                f"layer={runtime.layer_id}, wave={wave.wave_id}"
            )
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


class CudaMainSlotPool:
    def __init__(
        self,
        *,
        device: torch.device,
        num_slots: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
    ):
        self.device = device
        self.w13 = torch.empty((num_slots, *w13_shape), dtype=dtype, device=device)
        self.w2 = torch.empty((num_slots, *w2_shape), dtype=dtype, device=device)
        self.transfer_engine = CudaTransferEngine(device)
        self.owner: CudaLayerRuntime | None = None

    def acquire(self, runtime: CudaLayerRuntime, stream: torch.cuda.Stream) -> None:
        if self.owner is runtime:
            runtime._release_pending_to_stream(stream)
            return
        if self.owner is not None:
            self.owner._release_pending_to_stream(stream)
        runtime.invalidate_main_slots()
        self.owner = runtime


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
        main_slot_pool: CudaMainSlotPool | None = None,
        stage_pool: CudaStagePool | None = None,
        event_writer: JsonlEventWriter | None = None,
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
        self.main_slot_pool = main_slot_pool or CudaMainSlotPool(
            device=device,
            num_slots=num_slots,
            w13_shape=tuple(self.host_w13.shape[1:]),
            w2_shape=tuple(self.host_w2.shape[1:]),
            dtype=self.host_w13.dtype,
        )
        w13_parameter.data = self.main_slot_pool.w13
        w2_parameter.data = self.main_slot_pool.w2
        self.slot_w13_parameter = w13_parameter
        self.slot_w2_parameter = w2_parameter
        self.slot_w13 = w13_parameter
        self.slot_w2 = w2_parameter
        self.log2phy = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        self.expert_map = self.log2phy
        if "_expert_map" in experts_module._buffers:
            experts_module._buffers["_expert_map"] = self.expert_map
        else:
            if hasattr(experts_module, "_expert_map"):
                delattr(experts_module, "_expert_map")
            experts_module.register_buffer(
                "_expert_map", self.expert_map, persistent=False
            )
        if not hasattr(type(experts_module), "expert_map"):
            experts_module.expert_map = self.expert_map
        if hasattr(experts_module, "local_num_experts"):
            experts_module.local_num_experts = num_slots
        if hasattr(experts_module, "n_local_physical_experts"):
            experts_module.n_local_physical_experts = num_slots
        self.bank = ExpertSlotBank(num_slots)
        self.policy = LruPolicy()
        self.counters = RuntimeCounters()
        self.transfer_engine = self.main_slot_pool.transfer_engine
        self.stage_pool = stage_pool or CudaStagePool(
            device=device,
            num_slots=num_slots,
            w13_shape=tuple(self.host_w13.shape[1:]),
            w2_shape=tuple(self.host_w2.shape[1:]),
            dtype=self.host_w13.dtype,
        )
        self.last_wave_trace: WaveExecutionTrace | None = None
        self.event_writer = event_writer
        self.mapping_version = 0
        self._active_compute: ComputeHandle | None = None
        self._pending_computes: list[PendingCompute] = []
        self._pending_map_copies: list[PendingMapCopy] = []
        self._stable_ptrs = self.data_ptrs()

    @property
    def pending_map_copy_count(self) -> int:
        return len(self._pending_map_copies)

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

    def invalidate_main_slots(self) -> None:
        if self._active_compute is not None or self._pending_computes:
            raise RuntimeError(
                f"layer {self.layer_id} cannot invalidate slots with pending compute"
            )
        for slot in self.bank.slots:
            if slot.state is SlotState.READY:
                self.bank.evict(slot.slot_id)
            elif slot.state is not SlotState.EMPTY:
                raise RuntimeError(
                    f"layer {self.layer_id} cannot invalidate slot {slot.slot_id} "
                    f"while {slot.state.value}"
                )
        self.log2phy.fill_(-1)
        self.mapping_version += 1
        self.assert_stable_addresses()

    def _normalize_active(self, active_experts: Iterable[int]) -> tuple[int, ...]:
        active = tuple(dict.fromkeys(int(value) for value in active_experts))
        invalid = tuple(
            value for value in active if value < 0 or value >= self.num_experts
        )
        if invalid:
            raise ValueError(f"layer {self.layer_id} has invalid expert ids: {invalid}")
        if len(active) > self.num_slots:
            raise ActiveExpertCapacityError(
                f"layer={self.layer_id}, active_count={len(active)}, "
                f"slot_count={self.num_slots}"
            )
        return active

    def stage_sync(self, active_experts: Iterable[int]) -> LayerMappingSnapshot:
        self.main_slot_pool.acquire(
            self, torch.cuda.current_stream(self.slot_w13.device)
        )
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
                self.host_w13[expert_id].numel() * self.host_w13.element_size()
                + self.host_w2[expert_id].numel() * self.host_w2.element_size(),
            )

        snapshot = self._publish_mapping(active, selected, non_blocking=False)
        self.validate_snapshot(snapshot)
        self.assert_stable_addresses()
        return snapshot

    def stage_async(self, active_experts: Iterable[int]) -> LayerMappingSnapshot:
        if torch.cuda.is_current_stream_capturing():
            raise StagingDuringCaptureError(
                f"dynamic staging attempted during CUDA Graph capture: "
                f"layer={self.layer_id}"
            )
        self.main_slot_pool.acquire(self, self.transfer_engine.stream)
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
                    self.host_w13[copy.expert_id].numel() * self.host_w13.element_size()
                    + self.host_w2[copy.expert_id].numel()
                    * self.host_w2.element_size(),
                )

        snapshot = self._publish_mapping(active, selected, non_blocking=True)
        self.validate_snapshot(snapshot)
        self.assert_stable_addresses()
        return snapshot

    def _publish_mapping(
        self,
        active: tuple[int, ...],
        selected: dict[int, SlotLease],
        *,
        non_blocking: bool,
    ) -> LayerMappingSnapshot:
        self._pending_map_copies = [
            pending for pending in self._pending_map_copies if not pending.event.query()
        ]
        cpu_map = torch.full(
            (self.num_experts,),
            -1,
            dtype=torch.int32,
            device="cpu",
            pin_memory=True,
        )
        for key, slot_id in self.bank.ready_mapping().items():
            if key.layer_id == self.layer_id:
                cpu_map[key.expert_id] = slot_id
        self.log2phy.copy_(cpu_map, non_blocking=non_blocking)
        if non_blocking:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.log2phy.device))
            self._pending_map_copies.append(PendingMapCopy(cpu_map, event))
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
        if snapshot.mapping_version != self.mapping_version:
            raise StaleMappingError(
                f"stale mapping version: layer={self.layer_id}, "
                f"expected={self.mapping_version}, actual={snapshot.mapping_version}"
            )
        for expert_id, lease in zip(snapshot.active_experts, snapshot.leases):
            self.bank.validate_generation(lease.slot_id, lease.generation)
            if lease.key != ExpertKey(self.layer_id, expert_id):
                raise StaleMappingError(
                    f"stale expert lease: layer={self.layer_id}, expert={expert_id}, "
                    f"lease={lease.key}"
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
        self._release_pending_to_stream(self.transfer_engine.stream)

    def _release_pending_to_stream(self, stream: torch.cuda.Stream) -> None:
        for pending in self._pending_computes:
            stream.wait_event(pending.event)
            self.bank.end_compute(pending.handle)
        self._pending_computes.clear()

    def acquire_main_slots_for_current_stream(self) -> None:
        self.main_slot_pool.acquire(
            self, torch.cuda.current_stream(self.slot_w13.device)
        )
