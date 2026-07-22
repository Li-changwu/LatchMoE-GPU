import importlib.util
import json
from pathlib import Path


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "benchmark"
    / "scripts"
    / "analyze_cuda_timeline.py"
)
SPEC = importlib.util.spec_from_file_location("analyze_cuda_timeline", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_device_execution_excludes_memcpy_and_memset(tmp_path: Path):
    trace_path = tmp_path / "trace.json"
    trace_path.write_text(
        json.dumps(
            {
                "traceEvents": [
                    {
                        "ph": "X",
                        "cat": "gpu_user_annotation",
                        "name": "execute_context_0(0)_generation_8(8)",
                        "ts": 0,
                        "dur": 10_000,
                    },
                    {"ph": "X", "cat": "kernel", "ts": 1_000, "dur": 2_000},
                    {"ph": "X", "cat": "kernel", "ts": 2_000, "dur": 2_000},
                    {
                        "ph": "X",
                        "cat": "gpu_memcpy",
                        "ts": 4_000,
                        "dur": 1_000,
                    },
                    {
                        "ph": "X",
                        "cat": "gpu_memset",
                        "ts": 5_000,
                        "dur": 1_000,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )

    result = MODULE.analyze_trace(trace_path, average_tpot_ms=10.0)

    assert result["metrics"]["device_execution_ms"]["mean"] == 3.0
    assert result["metrics"]["host_induced_device_gaps_ms"]["mean"] == 7.0
    assert result["iterations"][0]["event_counts"] == {
        "gpu_memcpy": 1,
        "gpu_memset": 1,
        "kernel": 2,
    }
