#!/usr/bin/env python3
"""Decompose vLLM decode iterations from a PyTorch CUDA timeline."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


ANNOTATION = re.compile(
    r"^execute_context_(?P<context_requests>\d+)\((?P<context_tokens>\d+)\)"
    r"_generation_(?P<generation_requests>\d+)\((?P<generation_tokens>\d+)\)$"
)
DEVICE_EXECUTION_CATEGORY = "kernel"
GPU_ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


def interval_union_us(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    intervals.sort()
    start, end = intervals[0]
    total = 0.0
    for next_start, next_end in intervals[1:]:
        if next_start <= end:
            end = max(end, next_end)
        else:
            total += end - start
            start, end = next_start, next_end
    return total + end - start


def analyze_trace(path: Path, average_tpot_ms: float) -> dict[str, object]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]
    gpu_activities = [
        event
        for event in events
        if event.get("ph") == "X"
        and event.get("cat") in GPU_ACTIVITY_CATEGORIES
        and float(event.get("dur", 0)) > 0
    ]
    iterations = []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "gpu_user_annotation":
            continue
        match = ANNOTATION.match(str(event.get("name", "")))
        if not match:
            continue
        fields = {key: int(value) for key, value in match.groupdict().items()}
        if fields["context_requests"] != 0 or fields["generation_requests"] == 0:
            continue
        outer_start = float(event["ts"])
        outer_end = outer_start + float(event["dur"])
        intervals = []
        event_counts = {category: 0 for category in sorted(GPU_ACTIVITY_CATEGORIES)}
        for item in gpu_activities:
            item_start = float(item["ts"])
            item_end = item_start + float(item["dur"])
            if item_start >= outer_end or item_end <= outer_start:
                continue
            event_counts[item["cat"]] += 1
            if item["cat"] == DEVICE_EXECUTION_CATEGORY:
                intervals.append(
                    (max(outer_start, item_start), min(outer_end, item_end))
                )
        device_us = interval_union_us(intervals)
        device_ms = device_us / 1000.0
        iterations.append(
            {
                **fields,
                "gpu_annotation_duration_ms": float(event["dur"]) / 1000.0,
                "device_execution_ms": device_ms,
                "host_induced_device_gaps_ms": average_tpot_ms - device_ms,
                "event_counts": event_counts,
            }
        )
    if not iterations:
        raise ValueError(f"no pure decode iterations found in {path}")

    def stats(key: str) -> dict[str, float]:
        values = [float(item[key]) for item in iterations]
        ordered = sorted(values)
        return {
            "mean": statistics.fmean(values),
            "median": statistics.median(values),
            "p90": ordered[min(len(ordered) - 1, int(0.9 * len(ordered)))],
            "min": min(values),
            "max": max(values),
        }

    return {
        "trace": str(path.resolve()),
        "definition": {
            "device_execution": (
                "union of kernel intervals inside each pure-decode GPU "
                "execute_context range; excludes gpu_memcpy and gpu_memset"
            ),
            "host_induced_device_gaps": (
                "average decode TPOT minus kernel-only device execution; includes "
                "gpu_memcpy, gpu_memset, host scheduling, operator dispatch, runtime "
                "synchronization, graph replay launch, and resulting device idle time"
            ),
            "average_tpot": (
                "client mean_tpot_ms; both decomposition components cover decode TPOT "
                "only and exclude prefill and TTFT"
            ),
            "decode_filter": "context_requests == 0 and generation_requests > 0",
        },
        "decode_iterations": len(iterations),
        "generation_batch_sizes": sorted(
            {int(item["generation_requests"]) for item in iterations}
        ),
        "average_tpot_ms": average_tpot_ms,
        "metrics": {
            key: stats(key)
            for key in (
                "device_execution_ms",
                "host_induced_device_gaps_ms",
            )
        },
        "iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--average-tpot-ms",
        type=float,
        required=True,
        help="mean decode TPOT from the matching client result",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = analyze_trace(args.trace, args.average_tpot_ms)
    payload = json.dumps(result, indent=2) + "\n"
    if args.output:
        args.output.write_text(payload, encoding="utf-8")
    else:
        print(payload, end="")


if __name__ == "__main__":
    main()
