from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

from ..errors import PairIntegrityError
from ..routing import active_experts_from_topk


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


@dataclass(frozen=True)
class DeviceWaveDescriptor:
    wave_id: int
    experts: tuple[int, ...]
    pair_offsets: Any
    token_indices: Any
    logical_ids: Any
    physical_ids: Any
    pair_weights: Any
    expert_map: Any


@dataclass(frozen=True)
class DeviceExactWavePlan:
    capacity: int
    top_k: int
    num_tokens: int
    waves: tuple[DeviceWaveDescriptor, ...]
    compute_order: tuple[int, ...]

    @property
    def pair_count(self) -> int:
        return sum(int(wave.pair_offsets.numel()) for wave in self.waves)


@dataclass(frozen=True)
class MainCacheWaveSpec:
    """A deterministic serial wave for the persistent per-layer cache."""

    wave_id: int
    wave_type: Literal["hit", "miss"]
    experts: tuple[int, ...]
    overlap_candidate: bool = False

    @property
    def is_hit(self) -> bool:
        return self.wave_type == "hit"


def plan_main_cache_waves(
    active_experts: Iterable[int],
    capacity: int,
    *,
    hit_experts: Iterable[int] = (),
) -> tuple[MainCacheWaveSpec, ...]:
    """Plan hit-first waves without changing router-produced expert pairs.

    The hit set is emitted as one wave. Misses are then split into complete
    capacity-bounded waves in first-seen order. This makes the planner useful
    for both CPU tests and CUDA execution, while keeping duplicate routed pairs
    outside this expert-set planner.
    """
    if capacity <= 0:
        raise ValueError("capacity must be positive")
    active = tuple(dict.fromkeys(int(value) for value in active_experts))
    hits = set(int(value) for value in hit_experts)
    if any(value < 0 for value in active):
        raise ValueError("expert ids must be non-negative")
    if not hits.issubset(active):
        raise ValueError("hit_experts must be a subset of active_experts")
    specs: list[MainCacheWaveSpec] = []
    wave_id = 0
    hit_wave = tuple(value for value in active if value in hits)
    if hit_wave:
        specs.append(MainCacheWaveSpec(wave_id, "hit", hit_wave))
        wave_id += 1
    misses = tuple(value for value in active if value not in hits)
    for start in range(0, len(misses), capacity):
        specs.append(
            MainCacheWaveSpec(
                wave_id,
                "miss",
                misses[start : start + capacity],
            )
        )
        wave_id += 1
    if not specs and active:
        raise PairIntegrityError("active experts produced no cache waves")
    planned: list[MainCacheWaveSpec] = []
    for index, spec in enumerate(specs):
        next_spec = specs[index + 1] if index + 1 < len(specs) else None
        candidate = bool(
            next_spec is not None
            and 0 < len(spec.experts) < capacity
            and len(next_spec.experts) <= capacity - len(spec.experts)
        )
        planned.append(replace(spec, overlap_candidate=candidate))
    return tuple(planned)


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


def plan_device_exact_waves(
    topk_ids,
    topk_weights,
    *,
    capacity: int,
    num_experts: int,
    active_experts: Iterable[int] | None = None,
) -> DeviceExactWavePlan:
    """Build pair descriptors with device tensor operations.

    Only the unique expert list crosses to the host because H2D staging needs
    host-side source indices. Routed pair ids and weights remain on device.
    """
    import torch

    if capacity <= 0:
        raise ValueError("capacity must be positive")
    if num_experts <= 0:
        raise ValueError("num_experts must be positive")
    if topk_ids.ndim != 2 or topk_ids.numel() == 0:
        raise ValueError("routing tensors must be non-empty rank-2 tensors")
    if topk_ids.shape != topk_weights.shape:
        raise ValueError("topk_ids and topk_weights must have the same shape")
    if topk_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("topk_ids must use an integer dtype")

    if active_experts is None:
        active = active_experts_from_topk(topk_ids)
    else:
        active = tuple(sorted({int(value) for value in active_experts}))
    invalid = tuple(expert for expert in active if expert < 0 or expert >= num_experts)
    if invalid:
        raise ValueError(f"invalid expert ids: {invalid}")

    top_k = int(topk_ids.shape[1])
    flat_ids = topk_ids.reshape(-1).long()
    flat_weights = topk_weights.reshape(-1)
    waves: list[DeviceWaveDescriptor] = []
    for wave_id, start in enumerate(range(0, len(active), capacity)):
        experts = active[start : start + capacity]
        expert_tensor = torch.tensor(experts, dtype=torch.long, device=topk_ids.device)
        expert_map = torch.full(
            (num_experts,), -1, dtype=torch.int32, device=topk_ids.device
        )
        expert_map[expert_tensor] = torch.arange(
            len(experts), dtype=torch.int32, device=topk_ids.device
        )
        flat_physical = expert_map[flat_ids]
        pair_offsets = torch.nonzero(flat_physical >= 0, as_tuple=False).flatten()
        waves.append(
            DeviceWaveDescriptor(
                wave_id=wave_id,
                experts=experts,
                pair_offsets=pair_offsets,
                token_indices=torch.div(pair_offsets, top_k, rounding_mode="floor"),
                logical_ids=flat_ids.index_select(0, pair_offsets).reshape(-1, 1),
                physical_ids=(
                    flat_physical.index_select(0, pair_offsets).long().reshape(-1, 1)
                ),
                pair_weights=flat_weights.index_select(0, pair_offsets).reshape(-1, 1),
                expert_map=expert_map,
            )
        )

    order = tuple(range(len(waves)))
    plan = DeviceExactWavePlan(
        capacity=capacity,
        top_k=top_k,
        num_tokens=int(topk_ids.shape[0]),
        waves=tuple(waves),
        compute_order=order,
    )
    if plan.pair_count != int(topk_ids.numel()):
        raise PairIntegrityError(
            f"device pair coverage mismatch: expected={topk_ids.numel()}, "
            f"actual={plan.pair_count}"
        )
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
