#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from vllm_latchmoe_cuda.correctness import compare_greedy_results


def _read(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _run_manifest(path: Path) -> dict[str, Any]:
    return _read(path.parent / "run_manifest.json")


def _telemetry(path: Path) -> dict[str, Any]:
    return _read(path.parent / "offload_telemetry.json")


def _actual_overlap_events(path: Path) -> int:
    profile = path.parent / "profile.jsonl"
    return sum(
        json.loads(line).get("event") == "main_cache_overlap"
        and json.loads(line).get("actual_overlap") is True
        for line in profile.read_text(encoding="utf-8").splitlines()
        if line
    )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify exact-UVA, serial LatchMoE, and async LatchMoE token parity"
    )
    parser.add_argument("--uva", type=Path, required=True)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--async-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    uva = _read(args.uva)
    serial = _read(args.serial)
    async_result = _read(args.async_result)
    serial_comparison = compare_greedy_results(uva, serial)
    async_comparison = compare_greedy_results(uva, async_result)
    manifests = [_run_manifest(path) for path in (args.uva, args.serial, args.async_result)]
    source_states = [item.get("source_state_sha256") for item in manifests]
    telemetry = [_telemetry(path) for path in (args.uva, args.serial, args.async_result)]
    offload_bytes = [item.get("actual_offload_bytes") for item in telemetry]
    request_count = len(uva.get("outputs", []))
    actual_overlap_events = _actual_overlap_events(args.async_result)
    actual_overlap = actual_overlap_events > 0
    passed = (
        request_count == 4
        and serial_comparison.get("match") is True
        and async_comparison.get("match") is True
        and len(set(source_states)) == 1
        and None not in source_states
        and len(set(offload_bytes)) == 1
        and None not in offload_bytes
        and actual_overlap
    )
    report = {
        "schema_version": 1,
        "pass": passed,
        "qualification": "three_way_4_of_4_exact_token_parity" if passed else "failed",
        "request_count": request_count,
        "exact_uva_vs_serial": serial_comparison,
        "exact_uva_vs_async": async_comparison,
        "source_state_sha256": source_states[0] if len(set(source_states)) == 1 else None,
        "source_state_match": len(set(source_states)) == 1 and None not in source_states,
        "actual_offload_bytes": offload_bytes[0] if len(set(offload_bytes)) == 1 else None,
        "offload_bytes_match": len(set(offload_bytes)) == 1 and None not in offload_bytes,
        "async_actual_overlap": actual_overlap,
        "async_actual_overlap_events": actual_overlap_events,
        "inputs": {
            "exact_uva": str(args.uva),
            "serial_latchmoe": str(args.serial),
            "async_latchmoe": str(args.async_result),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("PASS: 4/4 three-way exact-token gate" if passed else "FAIL: exact-token gate")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
