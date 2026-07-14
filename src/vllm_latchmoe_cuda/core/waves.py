from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping, Sequence

from ..errors import PairIntegrityError


@dataclass(frozen=True)
class PairDescriptor:
    pair_offset: int
    token_index: int
    topk_position: int
    expert_id: int
    weight: float


@dataclass(frozen=True)
class WaveDescriptor:
    wave_id: int
    experts: tuple[int, ...]
    pairs: tuple[PairDescriptor, ...]


@dataclass(frozen=True)
class ExactWavePlan:
    capacity: int
    top_k: int
    num_tokens: int
    waves: tuple[WaveDescriptor, ...]
    compute_order: tuple[int, ...]
    issue_order: tuple[int, ...]

    def all_pair_offsets(self) -> tuple[int, ...]:
        return tuple(
            sorted(pair.pair_offset for wave in self.waves for pair in wave.pairs)
        )

    def pair(self, pair_offset: int) -> PairDescriptor:
        for wave in self.waves:
            for pair in wave.pairs:
                if pair.pair_offset == pair_offset:
                    return pair
        raise KeyError(f"pair_offset={pair_offset} is not in the wave plan")

    def with_issue_order(self, issue_order: Sequence[int]) -> ExactWavePlan:
        return replace(self, issue_order=tuple(int(value) for value in issue_order))


def _as_nested_list(value) -> list[list[float | int]]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [list(row) for row in value]


def plan_exact_waves(topk_ids, topk_weights, *, capacity: int) -> ExactWavePlan:
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    ids = _as_nested_list(topk_ids)
    weights = _as_nested_list(topk_weights)
    if not ids or not ids[0]:
        raise ValueError("routing tensors must be non-empty")
    if len(ids) != len(weights) or any(
        len(id_row) != len(weight_row) for id_row, weight_row in zip(ids, weights)
    ):
        raise ValueError("topk_ids and topk_weights must have the same shape")
    top_k = len(ids[0])
    if any(len(row) != top_k for row in ids):
        raise ValueError("routing tensors must be rectangular")

    active = sorted({int(expert) for row in ids for expert in row})
    if active[0] < 0:
        raise ValueError("expert ids must be non-negative")
    expert_waves = tuple(
        tuple(active[start : start + capacity])
        for start in range(0, len(active), capacity)
    )
    wave_by_expert = {
        expert: wave_id
        for wave_id, experts in enumerate(expert_waves)
        for expert in experts
    }
    pairs_by_wave: list[list[PairDescriptor]] = [list() for _ in expert_waves]
    for token_index, (id_row, weight_row) in enumerate(zip(ids, weights)):
        for topk_position, (expert, weight) in enumerate(zip(id_row, weight_row)):
            pair_offset = token_index * top_k + topk_position
            descriptor = PairDescriptor(
                pair_offset=pair_offset,
                token_index=token_index,
                topk_position=topk_position,
                expert_id=int(expert),
                weight=float(weight),
            )
            pairs_by_wave[wave_by_expert[int(expert)]].append(descriptor)

    waves = tuple(
        WaveDescriptor(wave_id, experts, tuple(pairs_by_wave[wave_id]))
        for wave_id, experts in enumerate(expert_waves)
    )
    order = tuple(range(len(waves)))
    plan = ExactWavePlan(capacity, top_k, len(ids), waves, order, order)
    validate_pair_coverage(plan, expected_pairs=len(ids) * top_k)
    return plan


def validate_pair_coverage(plan: ExactWavePlan, expected_pairs: int) -> None:
    offsets = [pair.pair_offset for wave in plan.waves for pair in wave.pairs]
    seen: set[int] = set()
    duplicate_values: set[int] = set()
    for value in offsets:
        if value in seen:
            duplicate_values.add(value)
        else:
            seen.add(value)
    duplicates = sorted(duplicate_values)
    if duplicates:
        raise PairIntegrityError(f"duplicate pair offsets: {duplicates}")
    actual = set(offsets)
    expected = set(range(expected_pairs))
    missing = sorted(expected - actual)
    extra = sorted(actual - expected)
    if missing or extra:
        raise PairIntegrityError(
            f"pair coverage mismatch: missing={missing}, extra={extra}"
        )
    for wave in plan.waves:
        if len(wave.experts) > plan.capacity:
            raise PairIntegrityError(
                f"wave {wave.wave_id} exceeds capacity {plan.capacity}"
            )
        expert_set = set(wave.experts)
        if any(pair.expert_id not in expert_set for pair in wave.pairs):
            raise PairIntegrityError(
                f"wave {wave.wave_id} contains a pair for an unassigned expert"
            )


def plan_transfer_issue_order(
    waves: Iterable[WaveDescriptor],
    *,
    ready_experts: frozenset[int],
    h2d_bytes_by_wave: Mapping[int, int],
) -> tuple[int, ...]:
    def priority(wave: WaveDescriptor) -> tuple[int, int, int]:
        missing = sum(expert not in ready_experts for expert in wave.experts)
        return (-int(h2d_bytes_by_wave.get(wave.wave_id, 0)), -missing, wave.wave_id)

    return tuple(wave.wave_id for wave in sorted(waves, key=priority))
