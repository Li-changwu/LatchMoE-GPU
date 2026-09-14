#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_latchmoe_cuda.benchmark import compare_mode_summaries


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare repeated LatchMoE ShareGPT metrics with official UVA"
    )
    parser.add_argument("--uva-summary", type=Path, required=True)
    parser.add_argument("--latchmoe-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-offload-mismatch",
        action="store_true",
        help="write a non-comparable report when actual offload bytes differ",
    )
    parser.add_argument(
        "--allow-contract-mismatch",
        action="store_true",
        help="write a diagnostic report when source/contract identities differ",
    )
    args = parser.parse_args()
    uva = json.loads(args.uva_summary.read_text(encoding="utf-8"))
    latchmoe = json.loads(args.latchmoe_summary.read_text(encoding="utf-8"))
    comparison = compare_mode_summaries(
        uva,
        latchmoe,
        require_equal_offload_bytes=not args.allow_offload_mismatch,
        require_equal_contract=not args.allow_contract_mismatch,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
