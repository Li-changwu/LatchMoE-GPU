import pytest
import torch

from vllm_latchmoe_cuda.graph_checks import (
    run_synthetic_graph_probe,
    validate_graph_report,
)


pytestmark = [
    pytest.mark.cuda,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.mark.parametrize(
    ("mode", "captured_segments"), [("eager", 0), ("piecewise", 1)]
)
def test_synthetic_graph_probe_records_real_boundary_checks(mode, captured_segments):
    report = run_synthetic_graph_probe(mode)

    validate_graph_report(report)
    assert report["scope"] == "synthetic-runtime"
    assert report["mode"] == mode
    assert report["captured_segment_count"] == captured_segments
    assert report["addresses_stable"] is True
    assert report["staging_during_capture_rejected"] is True
    assert report["output_close"] is True
