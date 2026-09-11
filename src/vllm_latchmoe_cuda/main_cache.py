"""Per-layer persistent CUDA main-slot cache ownership."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal, Mapping

import torch

from .core.expert_key import ExpertKey
from .core.policy import LruPolicy
from .core.slots import ExpertSlotBank, SlotLease, SlotState
from .core.waves import MainCacheWaveSpec
from .errors import StagingDuringCaptureError, StaleMappingError
from .transfer import CudaTransferEngine, ExpertCopy, TransferTicket


@dataclass(frozen=True)
class MainCacheIdentity:
    layer_id: int
    slot_w13_ptr: int
    slot_w2_ptr: int
    log2phy_ptr: int


@dataclass(frozen=True)
class PreparedMainCacheWave:
    layer_id: int
    wave_id: int
    wave_type: Literal["hit", "miss"]
    experts: tuple[int, ...]
    leases: tuple[SlotLease, ...]
    physical_slot_by_expert: Mapping[int, int]
    ready_ticket: TransferTicket | None
    evicted_experts: tuple[int, ...]
    h2d_bytes: int


class CudaLayerMainCache:
    """Stable storage and lifecycle state owned by exactly one MoE layer."""

    def __init__(
        self,
        *,
        layer_id: int,
        num_experts: int,
        num_slots: int,
        w13_shape: tuple[int, ...],
        w2_shape: tuple[int, ...],
        dtype: torch.dtype,
        device: torch.device,
    ):
        if device.type != "cuda":
            raise ValueError(f"main cache requires CUDA, got {device}")
        if not 0 < num_slots <= num_experts:
            raise ValueError("num_slots must be in [1, num_experts]")
        self.layer_id = int(layer_id)
        self.num_experts = int(num_experts)
        self.num_slots = int(num_slots)
        self.slot_w13 = torch.empty((num_slots, *w13_shape), dtype=dtype, device=device)
        self.slot_w2 = torch.empty((num_slots, *w2_shape), dtype=dtype, device=device)
        self.log2phy = torch.full((num_experts,), -1, dtype=torch.int32, device=device)
        self.bank = ExpertSlotBank(num_slots)
        self.policy = LruPolicy()
        self.transfer_engine = CudaTransferEngine(device)
        self.identity = MainCacheIdentity(
            self.layer_id,
            self.slot_w13.data_ptr(),
            self.slot_w2.data_ptr(),
            self.log2phy.data_ptr(),
        )

    def data_ptrs(self) -> dict[str, int]:
        return {
            "slot_w13": self.slot_w13.data_ptr(),
            "slot_w2": self.slot_w2.data_ptr(),
            "log2phy": self.log2phy.data_ptr(),
        }

    def ready_mapping(self) -> Mapping[int, int]:
        return {
            key.expert_id: slot_id
            for key, slot_id in self.bank.ready_mapping().items()
            if key.layer_id == self.layer_id
        }

    def lease_for(self, expert_id: int) -> SlotLease | None:
        slot = self.bank.find(ExpertKey(self.layer_id, int(expert_id)))
        if slot is None or slot.key is None:
            return None
        return SlotLease(slot.slot_id, slot.key, slot.generation)

    def prepare_wave(
        self,
        spec: MainCacheWaveSpec,
        *,
        host_w13: torch.Tensor,
        host_w2: torch.Tensor,
        protected_slots: frozenset[int] = frozenset(),
    ) -> PreparedMainCacheWave:
        """Resolve hits and replace only evictable slots in this layer cache."""
        if torch.cuda.is_current_stream_capturing():
            raise StagingDuringCaptureError(
                f"main-cache staging attempted during CUDA Graph capture: layer={self.layer_id}"
            )
        selected: dict[int, SlotLease] = {}
        reserved = set(protected_slots)
        evicted: list[int] = []
        copies: list[ExpertCopy] = []
        for expert_id in spec.experts:
            lease = self.lease_for(expert_id)
            if lease is not None:
                selected[expert_id] = lease
                reserved.add(lease.slot_id)
        if spec.wave_type == "hit" and len(selected) != len(spec.experts):
            missing = tuple(expert for expert in spec.experts if expert not in selected)
            raise StaleMappingError(
                f"hit wave lost READY experts before execution: layer={self.layer_id}, missing={missing}"
            )
        for expert_id in spec.experts:
            if expert_id in selected:
                continue
            slot_id = self.policy.choose(self.bank, excluded=reserved)
            slot = self.bank.slots[slot_id]
            if slot.state is SlotState.READY:
                if slot.key is not None:
                    evicted.append(slot.key.expert_id)
                self.bank.evict(slot_id)
            key = ExpertKey(self.layer_id, expert_id)
            lease = self.bank.begin_load(key, slot_id=slot_id)
            copies.append(ExpertCopy(expert_id, slot_id, lease.generation))
            selected[expert_id] = lease
            reserved.add(slot_id)
        ticket = None
        if copies:
            ticket = self.transfer_engine.load_many_async(
                host_w13=host_w13,
                host_w2=host_w2,
                slot_w13=self.slot_w13,
                slot_w2=self.slot_w2,
                copies=copies,
            )
        return PreparedMainCacheWave(
            layer_id=self.layer_id,
            wave_id=spec.wave_id,
            wave_type=spec.wave_type,
            experts=spec.experts,
            leases=tuple(selected[expert] for expert in spec.experts),
            physical_slot_by_expert=MappingProxyType(
                {expert: selected[expert].slot_id for expert in spec.experts}
            ),
            ready_ticket=ticket,
            evicted_experts=tuple(evicted),
            h2d_bytes=sum(
                host_w13[copy.expert_id].numel() * host_w13.element_size()
                + host_w2[copy.expert_id].numel() * host_w2.element_size()
                for copy in copies
            ),
        )

    def wait_and_publish(self, prepared: PreparedMainCacheWave) -> None:
        if prepared.layer_id != self.layer_id:
            raise ValueError("prepared wave belongs to a different layer")
        if prepared.ready_ticket is not None:
            self.transfer_engine.wait_ready(prepared.ready_ticket)
            for copy in prepared.ready_ticket.copies:
                self.bank.mark_ready(copy.slot_id, copy.generation)
        self.log2phy.fill_(-1)
        for key, slot_id in self.bank.ready_mapping().items():
            if key.layer_id == self.layer_id:
                self.log2phy[key.expert_id] = slot_id
