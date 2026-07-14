import subprocess
import sys
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.graph_checks import (
    GraphCheckError,
    validate_graph_report,
)


ROOT = Path(__file__).resolve().parents[2]


def _report(**updates):
    report = {
        "schema_version": 1,
        "scope": "synthetic-runtime",
        "mode": "piecewise",
        "addresses_stable": True,
        "staging_during_capture_rejected": True,
        "graph_break_count": 1,
        "captured_segment_count": 1,
        "allocated_growth_bytes": 0,
        "allocated_growth_limit_bytes": 1024 * 1024,
        "reserved_growth_bytes": 0,
        "reserved_growth_limit_bytes": 1024 * 1024,
    }
    report.update(updates)
    return report


def test_piecewise_report_requires_capture_and_stable_addresses():
    assert validate_graph_report(_report())["mode"] == "piecewise"

    with pytest.raises(GraphCheckError, match="captured segment"):
        validate_graph_report(_report(captured_segment_count=0))
    with pytest.raises(GraphCheckError, match="address"):
        validate_graph_report(_report(addresses_stable=False))


def test_eager_report_requires_graph_break_but_not_capture():
    report = _report(mode="eager", graph_break_count=1, captured_segment_count=0)

    assert validate_graph_report(report)["mode"] == "eager"

    with pytest.raises(GraphCheckError, match="graph break"):
        validate_graph_report(dict(report, graph_break_count=0))


def test_graph_report_rejects_allocator_growth_above_limit():
    with pytest.raises(GraphCheckError, match="allocator growth"):
        validate_graph_report(_report(allocated_growth_bytes=2 * 1024 * 1024))


def test_graph_runner_exposes_help():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_graph_checks.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--mode" in completed.stdout
    assert "--artifact-dir" in completed.stdout
