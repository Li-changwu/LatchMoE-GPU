#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from vllm_latchmoe_cuda.artifacts import ArtifactRun, RunKind
from vllm_latchmoe_cuda.graph_checks import (
    run_synthetic_graph_probe,
    validate_graph_report,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Record eager or PIECEWISE CUDA graph boundary checks"
    )
    parser.add_argument("--mode", choices=("eager", "piecewise"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--kind", choices=tuple(kind.value for kind in RunKind), default="smoke"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    run = ArtifactRun.create(
        args.artifact_dir,
        kind=RunKind(args.kind),
        command=[
            sys.executable,
            str(Path(__file__).resolve()),
            *(argv or sys.argv[1:]),
        ],
    )
    run.write_text("stdout.log", "")
    run.write_text("stderr.log", "")
    try:
        report = run_synthetic_graph_probe(args.mode)
        validate_graph_report(report)
        run.write_json("graph_checks.json", report)
        run.record_completion()
        return_code = 0
    except BaseException as error:
        run.record_failure(error)
        return_code = 1
    finally:
        run.write_inventory()
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
