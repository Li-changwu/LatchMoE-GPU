from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch


@dataclass(frozen=True)
class ExpertCopy:
    expert_id: int
    slot_id: int
    generation: int


@dataclass(frozen=True)
class TransferTicket:
    event: torch.cuda.Event
    copies: tuple[ExpertCopy, ...]
    start_event: torch.cuda.Event | None = None
    end_event: torch.cuda.Event | None = None
    h2d_bytes: int = 0


def contiguous_copy_runs(
    copies: Iterable[ExpertCopy],
) -> tuple[tuple[ExpertCopy, ...], ...]:
    values = tuple(copies)
    if not values:
        return ()
    runs: list[list[ExpertCopy]] = [[values[0]]]
    for copy in values[1:]:
        previous = runs[-1][-1]
        if (
            copy.expert_id == previous.expert_id + 1
            and copy.slot_id == previous.slot_id + 1
        ):
            runs[-1].append(copy)
        else:
            runs.append([copy])
    return tuple(tuple(run) for run in runs)


class CudaTransferEngine:
    def __init__(self, device: torch.device):
        self.device = device
        self.stream = torch.cuda.Stream(device=device)

    def load_many_async(
        self,
        *,
        host_w13: torch.Tensor,
        host_w2: torch.Tensor,
        slot_w13: torch.Tensor,
        slot_w2: torch.Tensor,
        copies: Iterable[ExpertCopy],
        origin_event: torch.cuda.Event | None = None,
    ) -> TransferTicket:
        copies = tuple(copies)
        with torch.cuda.stream(self.stream):
            if origin_event is not None:
                self.stream.wait_event(origin_event)
            start_event = torch.cuda.Event(enable_timing=True)
            start_event.record(self.stream)
            for run in contiguous_copy_runs(copies):
                source_start = run[0].expert_id
                slot_start = run[0].slot_id
                length = len(run)
                slot_w13.narrow(0, slot_start, length).copy_(
                    host_w13.narrow(0, source_start, length), non_blocking=True
                )
                slot_w2.narrow(0, slot_start, length).copy_(
                    host_w2.narrow(0, source_start, length), non_blocking=True
                )
            event = torch.cuda.Event(enable_timing=True)
            event.record(self.stream)
        return TransferTicket(
            event=event,
            copies=copies,
            start_event=start_event,
            end_event=event,
            h2d_bytes=sum(
                int(host_w13[copy.expert_id].numel() * host_w13.element_size())
                + int(host_w2[copy.expert_id].numel() * host_w2.element_size())
                for copy in copies
            ),
        )

    def wait_ready(self, ticket: TransferTicket) -> None:
        torch.cuda.current_stream(self.device).wait_event(ticket.event)

    def drain(self, ticket: TransferTicket) -> None:
        ticket.event.synchronize()

    def close(self) -> None:
        self.stream.synchronize()


def copy_expert_sync(
    *,
    host_w13: torch.Tensor,
    host_w2: torch.Tensor,
    expert_id: int,
    slot_w13: torch.Tensor,
    slot_w2: torch.Tensor,
    slot_id: int,
) -> None:
    slot_w13[slot_id].copy_(host_w13[expert_id], non_blocking=False)
    slot_w2[slot_id].copy_(host_w2[expert_id], non_blocking=False)
