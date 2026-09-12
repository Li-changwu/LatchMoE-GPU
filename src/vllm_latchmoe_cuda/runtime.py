from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import torch
from torch import nn

from .core.expert_key import ExpertKey
from .core.policy import LruPolicy
from .core.slots import ComputeHandle, SlotLease, SlotState
from .core.waves import MainCacheWaveSpec
from .errors import (
    ActiveExpertCapacityError,
    NoEvictableSlotError,
    StableAddressError,
    StagingDuringCaptureError,
    StaleMappingError,
    RuntimePoisonedError,
)
from .host_store import PinnedHostStore
from .main_cache import CudaLayerMainCache, PreparedMainCacheWave
from .manifest import LayerLayout
from .profile import JsonlEventWriter, RuntimeCounters
from .transfer import ExpertCopy, copy_expert_sync


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


class RuntimeState(str, Enum):
    ACTIVE = "ACTIVE"
    POISONED = "POISONED"
    CLOSED = "CLOSED"


@dataclass(frozen=True)
class WaveExecutionTrace:
    pair_count: int
    compute_order: tuple[int, ...]
    issue_order: tuple[int, ...]
    buffer_by_wave: tuple[tuple[int, int], ...]
class CudaLayerRuntime:
    def __init__(
        self,
        *,
        layer: LayerLayout | None = None,
        layer_id: int | None = None,
        num_experts: int,
        num_slots: int,
        host_store: PinnedHostStore,
        experts_module: nn.Module,
        device: torch.device,
        event_writer: JsonlEventWriter | None = None,
    ):
        if device.type != "cuda":
            raise ValueError(f"CUDA runtime requires a CUDA device, got {device}")
        if layer_id is None:
            if layer is None:
                raise TypeError("layer_id is required")
            layer_id = layer.layer_id
        self.layer_id = int(layer_id)
        self.device = device
        self.state = RuntimeState.ACTIVE
        self.poison_cause: BaseException | None = None
        self._closed = False
        self.production_plan = False
        self.router_call_count = 0
        self.num_experts = num_experts
        self.num_slots = num_slots
        self.host_w13 = host_store.tensor_view(self.layer_id, "w13_weight")
        self.host_w2 = host_store.tensor_view(self.layer_id, "w2_weight")
        w13_parameter = getattr(experts_module, "w13_weight")
        w2_parameter = getattr(experts_module, "w2_weight")
        self.main_cache = CudaLayerMainCache(
            layer_id=self.layer_id,
            device=device,
            num_slots=num_slots,
            num_experts=num_experts,
            w13_shape=tuple(self.host_w13.shape[1:]),
            w2_shape=tuple(self.host_w2.shape[1:]),
            dtype=self.host_w13.dtype,
        )
        w13_parameter.data = self.main_cache.slot_w13
        w2_parameter.data = self.main_cache.slot_w2
        self.slot_w13_parameter = w13_parameter
        self.slot_w2_parameter = w2_parameter
        self.slot_w13 = w13_parameter
        self.slot_w2 = w2_parameter
        self.log2phy = self.main_cache.log2phy
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
        self.bank = self.main_cache.bank
        self.policy = LruPolicy()
        self.counters = RuntimeCounters()
        self.transfer_engine = self.main_cache.transfer_engine
        self.last_wave_trace: WaveExecutionTrace | None = None
        self.event_writer = event_writer
        self.direct_slots_profiled = False
        self.mapping_version = 0
        self._active_compute: ComputeHandle | None = None
        self._main_cache_compute_handles: dict[int, ComputeHandle] = {}
        self._main_cache_windows: dict[int, dict[str, torch.cuda.Event | None]] = {}
        self._pending_main_computes: dict[int, PendingCompute] = {}
        self._pending_transfer_tickets: dict[int, object] = {}
        self.overlap_records: list[dict[str, object]] = []
        self._pending_computes: list[PendingCompute] = []
        self._pending_map_copies: list[PendingMapCopy] = []
        self._free_map_buffers = [self._new_cpu_map(), self._new_cpu_map()]
        self._graph_overflow_active: tuple[int, ...] | None = None
        self.graph_token = torch.zeros((), dtype=torch.int64, device=device)
        self._stable_ptrs = self.data_ptrs()

    @property
    def pending_map_copy_count(self) -> int:
        return len(self._pending_map_copies)

    @property
    def map_buffer_count(self) -> int:
        return len(self._free_map_buffers) + len(self._pending_map_copies)

    @property
    def is_poisoned(self) -> bool:
        return self.state is RuntimeState.POISONED

    def ensure_healthy(self) -> None:
        if self.state is RuntimeState.POISONED:
            raise RuntimePoisonedError(
                f"layer {self.layer_id} runtime is poisoned"
            ) from self.poison_cause
        if self.state is RuntimeState.CLOSED:
            raise RuntimePoisonedError(f"layer {self.layer_id} runtime is closed")

    @property
    def main_slot_pool(self):
        """Compatibility view; storage is owned by this layer's main cache."""
        return self.main_cache

    def _new_cpu_map(self) -> torch.Tensor:
        return torch.empty(
            (self.num_experts,), dtype=torch.int32, device="cpu", pin_memory=True
        )

    def data_ptrs(self) -> dict[str, int]:
        return {
            "slot_w13": self.slot_w13.data_ptr(),
            "slot_w2": self.slot_w2.data_ptr(),
            "log2phy": self.log2phy.data_ptr(),
            "expert_map": self.expert_map.data_ptr(),
            "graph_token": self.graph_token.data_ptr(),
        }

    def assert_stable_addresses(self) -> None:
        actual = self.data_ptrs()
        if actual != self._stable_ptrs:
            raise StableAddressError(
                f"layer {self.layer_id} tensor address changed: "
                f"expected={self._stable_ptrs}, actual={actual}"
            )

    def _normalize_active(
        self, active_experts: Iterable[int], *, enforce_capacity: bool = True
    ) -> tuple[int, ...]:
        active = tuple(dict.fromkeys(int(value) for value in active_experts))
        invalid = tuple(
            value for value in active if value < 0 or value >= self.num_experts
        )
        if invalid:
            raise ValueError(f"layer {self.layer_id} has invalid expert ids: {invalid}")
        if enforce_capacity and len(active) > self.num_slots:
            raise ActiveExpertCapacityError(
                f"layer={self.layer_id}, active_count={len(active)}, "
                f"slot_count={self.num_slots}"
            )
        return active

    @property
    def graph_overflow_active(self) -> tuple[int, ...] | None:
        return self._graph_overflow_active

    def prepare_graph_compute(self, active_experts: Iterable[int]) -> bool:
        """Stage a graph-safe working set or defer an overflow to exact waves."""
        self.ensure_healthy()
        active = self._normalize_active(active_experts, enforce_capacity=False)
        if len(active) > self.num_slots:
            if self._active_compute is not None:
                raise RuntimeError(f"layer {self.layer_id} already has active compute")
            self._graph_overflow_active = active
            return False
        self._graph_overflow_active = None
        self.prepare_compute_async(active)
        return True

    def finish_graph_compute(self) -> PendingCompute | None:
        if self._graph_overflow_active is not None:
            self._graph_overflow_active = None
            return None
        return self.finish_compute_async()

    def _choose_slot(self, expert_id: int, reserved: set[int]) -> int:
        if self.num_slots == self.num_experts:
            return expert_id
        return self.policy.choose(self.bank, excluded=reserved)

    def stage_sync(self, active_experts: Iterable[int]) -> LayerMappingSnapshot:
        self.ensure_healthy()
        self.release_pending_for_transfer()
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
            slot_id = self._choose_slot(expert_id, reserved)
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
        self.ensure_healthy()
        if torch.cuda.is_current_stream_capturing():
            raise StagingDuringCaptureError(
                f"dynamic staging attempted during CUDA Graph capture: "
                f"layer={self.layer_id}"
            )
        self.release_pending_for_transfer()
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
            slot_id = self._choose_slot(expert_id, reserved)
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
        still_pending: list[PendingMapCopy] = []
        for pending in self._pending_map_copies:
            if pending.event.query():
                self._free_map_buffers.append(pending.cpu_map)
            else:
                still_pending.append(pending)
        self._pending_map_copies = still_pending
        cpu_map = (
            self._free_map_buffers.pop()
            if self._free_map_buffers
            else self._new_cpu_map()
        )
        cpu_map.fill_(-1)
        for key, slot_id in self.bank.ready_mapping().items():
            if key.layer_id == self.layer_id:
                cpu_map[key.expert_id] = slot_id
        self.log2phy.copy_(cpu_map, non_blocking=non_blocking)
        if non_blocking:
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.log2phy.device))
            self._pending_map_copies.append(PendingMapCopy(cpu_map, event))
        else:
            self._free_map_buffers.append(cpu_map)
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
        self.ensure_healthy()
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
        self.ensure_healthy()
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
        self._release_pending_to_stream(
            torch.cuda.current_stream(self.slot_w13.device)
        )

    def prepare_main_cache_wave(
        self,
        spec: MainCacheWaveSpec,
        *,
        protected_slots: frozenset[int] = frozenset(),
        origin_event: torch.cuda.Event | None = None,
        overlap_candidate: bool | None = None,
    ) -> PreparedMainCacheWave:
        self.ensure_healthy()
        prepared = self.main_cache.prepare_wave(
            spec,
            host_w13=self.host_w13,
            host_w2=self.host_w2,
            protected_slots=protected_slots,
            origin_event=origin_event,
            overlap_candidate=overlap_candidate,
        )
        self.counters.increment("slot_hit", sum(
            1 for expert in spec.experts if self.main_cache.lease_for(expert) is not None
        ) if spec.wave_type == "hit" else 0)
        self.counters.increment("slot_miss", len(prepared.leases) if spec.wave_type == "miss" else 0)
        self.counters.increment("h2d_bytes", prepared.h2d_bytes)
        if prepared.ready_ticket is not None:
            self._pending_transfer_tickets[prepared.wave_id] = prepared.ready_ticket
        return prepared

    def wait_and_publish(self, prepared: PreparedMainCacheWave) -> None:
        self.ensure_healthy()
        self.main_cache.wait_and_publish(prepared)
        self._pending_transfer_tickets.pop(prepared.wave_id, None)
        self.mapping_version += 1
        self.assert_stable_addresses()

    def begin_main_cache_compute(self, prepared: PreparedMainCacheWave) -> ComputeHandle:
        self.ensure_healthy()
        handle = self.bank.begin_compute(tuple(lease.slot_id for lease in prepared.leases))
        self._main_cache_compute_handles[prepared.wave_id] = handle
        start_event = torch.cuda.Event(enable_timing=True)
        start_event.record(torch.cuda.current_stream(self.device))
        self._main_cache_windows[prepared.wave_id] = {
            "start": start_event,
            "end": None,
        }
        return handle

    def mark_compute_complete(
        self, prepared: PreparedMainCacheWave, *, defer: bool = False
    ) -> None:
        handle = self._main_cache_compute_handles.pop(prepared.wave_id, None)
        if handle is not None:
            end_event = torch.cuda.Event(enable_timing=True)
            end_event.record(torch.cuda.current_stream(self.device))
            window = self._main_cache_windows.setdefault(prepared.wave_id, {})
            window["end"] = end_event
            if defer:
                self._pending_main_computes[prepared.wave_id] = PendingCompute(
                    handle=handle, event=end_event
                )
            else:
                self.bank.end_compute(handle)

    def complete_pending_main_cache_computes(self) -> None:
        for wave_id, pending in tuple(self._pending_main_computes.items()):
            pending.event.synchronize()
            self.bank.end_compute(pending.handle)
            del self._pending_main_computes[wave_id]

    def compute_start_event(self, wave_id: int) -> torch.cuda.Event | None:
        window = self._main_cache_windows.get(wave_id)
        return None if window is None else window.get("start")

    def record_overlap(self, prepared: PreparedMainCacheWave) -> dict[str, object] | None:
        ticket = prepared.ready_ticket
        window = self._main_cache_windows.get(prepared.wave_id - 1)
        if ticket is None or window is None or not prepared.overlap_candidate:
            return None
        self.complete_pending_main_cache_computes()
        start = window.get("start")
        end = window.get("end")
        transfer_start = ticket.start_event
        transfer_end = ticket.end_event or ticket.event
        if start is None or end is None or transfer_start is None or transfer_end is None:
            return None
        transfer_end.synchronize()
        origin_ms = 0.0
        h2d_start_ms = origin_ms + float(start.elapsed_time(transfer_start))
        h2d_end_ms = origin_ms + float(start.elapsed_time(transfer_end))
        compute_start_ms = 0.0
        compute_end_ms = float(start.elapsed_time(end))
        actual = max(h2d_start_ms, compute_start_ms) < min(
            h2d_end_ms, compute_end_ms
        )
        record: dict[str, object] = {
            "event": "main_cache_overlap",
            "layer_id": self.layer_id,
            "wave_id": prepared.wave_id,
            "overlap_candidate": True,
            "actual_overlap": bool(actual),
            "h2d_start_ms": h2d_start_ms,
            "h2d_end_ms": h2d_end_ms,
            "compute_start_ms": compute_start_ms,
            "compute_end_ms": compute_end_ms,
            "h2d_bytes": prepared.h2d_bytes,
            "protected_slot_ids": list(prepared.protected_slots),
        }
        self.overlap_records.append(record)
        if self.event_writer is not None:
            fields = dict(record)
            fields.pop("event", None)
            self.event_writer.write("main_cache_overlap", **fields)
        return record

    def poison(
        self,
        cause: BaseException,
        *,
        wave_id: int | None = None,
        compute_done: torch.cuda.Event | None = None,
        transfer_tickets: tuple[object, ...] = (),
        active_experts: tuple[int, ...] = (),
    ) -> None:
        if self.state is RuntimeState.CLOSED:
            return
        if self.state is RuntimeState.POISONED:
            return
        self.state = RuntimeState.POISONED
        self.poison_cause = cause
        if compute_done is not None:
            self._main_cache_windows.setdefault(wave_id or -1, {})["end"] = compute_done
            if wave_id is not None:
                handle = self._main_cache_compute_handles.pop(wave_id, None)
                if handle is not None:
                    self._pending_main_computes[wave_id] = PendingCompute(
                        handle=handle, event=compute_done
                    )
        for ticket in transfer_tickets:
            if getattr(ticket, "event", None) is not None:
                self._pending_transfer_tickets[id(ticket)] = ticket
        self._failure_record = failure = {
            "event": "failure",
            "plan_id": getattr(self, "plan_id", None),
            "layer_id": self.layer_id,
            "wave_id": wave_id,
            "exception_type": type(cause).__name__,
            "exception": str(cause),
            "active_experts": list(active_experts),
            "leases": [
                {
                    "slot_id": slot.slot_id,
                    "expert_id": None if slot.key is None else slot.key.expert_id,
                    "generation": slot.generation,
                    "state": slot.state.value,
                }
                for slot in self.bank.slots
            ],
            "pending_compute_events": len(self._pending_main_computes),
            "pending_transfer_events": len(self._pending_transfer_tickets),
            "drained": False,
        }
        if self.event_writer is not None:
            try:
                fields = dict(failure)
                fields.pop("event", None)
                self.event_writer.write("failure", **fields)
            except BaseException:
                pass

    def close(self) -> None:
        if self._closed:
            return
        close_error: BaseException | None = None
        try:
            for pending in tuple(self._pending_main_computes.values()):
                pending.event.synchronize()
            self.complete_pending_main_cache_computes()
            for ticket in tuple(self._pending_transfer_tickets.values()):
                self.transfer_engine.drain(ticket)
            for pending in tuple(self._pending_map_copies):
                pending.event.synchronize()
            self.main_cache.close()
            if self.event_writer is not None and hasattr(self, "_failure_record"):
                drained = dict(self._failure_record)
                drained["drained"] = True
                drained["pending_compute_events"] = 0
                drained["pending_transfer_events"] = 0
                fields = dict(drained)
                fields.pop("event", None)
                try:
                    self.event_writer.write("failure", **fields)
                except BaseException:
                    pass
        except BaseException as exc:
            close_error = exc
        finally:
            self._closed = True
            self.state = RuntimeState.CLOSED
        if close_error is not None:
            raise close_error
