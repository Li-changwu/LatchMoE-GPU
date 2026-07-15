#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from vllm_latchmoe_cuda.benchmark import compare_ablation_summaries


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the strict UVA/LatchMoE x eager/PIECEWISE ablation"
    )
    parser.add_argument("--uva-eager-summary", type=Path, required=True)
    parser.add_argument("--uva-piecewise-summary", type=Path, required=True)
    parser.add_argument("--latchmoe-eager-summary", type=Path, required=True)
    parser.add_argument("--latchmoe-piecewise-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    comparison = compare_ablation_summaries(
        uva_eager=_read(args.uva_eager_summary),
        uva_piecewise=_read(args.uva_piecewise_summary),
        latchmoe_eager=_read(args.latchmoe_eager_summary),
        latchmoe_piecewise=_read(args.latchmoe_piecewise_summary),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(comparison, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(comparison, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
