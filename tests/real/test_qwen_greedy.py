import os
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.correctness import require_greedy_match_files


@pytest.mark.e2e
@pytest.mark.cuda
@pytest.mark.skipif(
    os.getenv("LATCHMOE_RUN_E2E") != "1",
    reason="set LATCHMOE_RUN_E2E=1 to run full Qwen greedy comparison",
)
def test_qwen_greedy_e2e_artifacts_have_identical_token_ids():
    reference = os.getenv("LATCHMOE_E2E_REFERENCE_JSON")
    candidate = os.getenv("LATCHMOE_E2E_CANDIDATE_JSON")
    assert reference, "LATCHMOE_E2E_REFERENCE_JSON must name the UVA result"
    assert candidate, "LATCHMOE_E2E_CANDIDATE_JSON must name the LatchMoE result"

    comparison = require_greedy_match_files(Path(reference), Path(candidate))

    assert comparison["match"] is True
