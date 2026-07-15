import json
import subprocess
import sys
from pathlib import Path

import pytest

from vllm_latchmoe_cuda.correctness import (
    STOCK_UVA_CPU_OFFLOAD_GB,
    CorrectnessMismatchError,
    build_engine_kwargs,
    compare_greedy_results,
    require_greedy_match,
    resolve_correctness_mode,
)


ROOT = Path(__file__).resolve().parents[2]


def _result(mode: str, token_ids: list[list[int]]) -> dict[str, object]:
    backend = "uva" if mode == "uva" else "latchmoe"
    return {
        "schema_version": 1,
        "mode": mode,
        "backend": backend,
        "enforce_eager": mode != "latchmoe-piecewise",
        "expected_wave_event": mode == "latchmoe-waves",
        "manifest_sha256": "a" * 64,
        "model_revision": "revision-1",
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "prompts": ["alpha", "beta"],
        "sampling": {"temperature": 0.0, "max_tokens": 16, "seed": 0},
        "outputs": [
            {
                "prompt": "alpha",
                "prompt_token_ids": [11, 12],
                "token_ids": token_ids[0],
                "text": "A",
            },
            {
                "prompt": "beta",
                "prompt_token_ids": [21, 22],
                "token_ids": token_ids[1],
                "text": "B",
            },
        ],
    }


@pytest.mark.parametrize(
    ("mode", "backend", "enforce_eager", "expect_waves"),
    [
        ("uva", "uva", True, False),
        ("latchmoe-eager", "latchmoe", True, False),
        ("latchmoe-piecewise", "latchmoe", False, False),
        ("latchmoe-waves", "latchmoe", True, True),
    ],
)
def test_correctness_mode_has_explicit_backend_and_graph_policy(
    mode, backend, enforce_eager, expect_waves
):
    config = resolve_correctness_mode(mode)

    assert config.backend == backend
    assert config.enforce_eager is enforce_eager
    assert config.expect_waves is expect_waves


def test_engine_kwargs_lock_target_and_piecewise_mode(tiny_manifest):
    kwargs = build_engine_kwargs(
        manifest=tiny_manifest,
        mode=resolve_correctness_mode("latchmoe-piecewise"),
        max_model_len=512,
        gpu_memory_utilization=0.98,
        kv_cache_memory_bytes=256 * 1024 * 1024,
    )

    assert kwargs["model"] == tiny_manifest.model.path
    assert kwargs["dtype"] == "bfloat16"
    assert kwargs["tensor_parallel_size"] == 1
    assert kwargs["enforce_eager"] is False
    assert kwargs["compilation_config"] == {"cudagraph_mode": "PIECEWISE"}


def test_engine_kwargs_enable_stock_uva_with_fixed_budget(tiny_manifest):
    kwargs = build_engine_kwargs(
        manifest=tiny_manifest,
        mode=resolve_correctness_mode("uva"),
        max_model_len=512,
        gpu_memory_utilization=0.98,
        kv_cache_memory_bytes=256 * 1024 * 1024,
    )

    assert kwargs["cpu_offload_gb"] == STOCK_UVA_CPU_OFFLOAD_GB == 14.0


def test_worker_environment_delegates_uva_to_stock_factory(monkeypatch):
    from scripts.run_correctness import _worker_environment

    monkeypatch.setenv("VLLM_LATCHMOE_MODE", "stale")
    monkeypatch.setenv("VLLM_LATCHMOE_MANIFEST", "stale")
    monkeypatch.setenv("VLLM_LATCHMOE_PROFILE_PATH", "stale")
    manifest_path = Path("tiny_manifest.json")
    profile_path = Path("profile.jsonl")

    uva = _worker_environment(
        resolve_correctness_mode("uva"), manifest_path, profile_path
    )
    latchmoe = _worker_environment(
        resolve_correctness_mode("latchmoe-eager"), manifest_path, profile_path
    )

    assert uva["VLLM_PLUGINS"] == "latchmoe_cuda"
    assert "VLLM_LATCHMOE_MODE" not in uva
    assert "VLLM_LATCHMOE_MANIFEST" not in uva
    assert "VLLM_LATCHMOE_PROFILE_PATH" not in uva
    assert latchmoe["VLLM_LATCHMOE_MODE"] == "latchmoe"
    assert latchmoe["VLLM_LATCHMOE_MANIFEST"] == str(manifest_path)
    assert latchmoe["VLLM_LATCHMOE_PROFILE_PATH"] == str(profile_path)


def test_greedy_comparison_requires_exact_token_ids():
    reference = _result("uva", [[1, 2, 3], [4, 5]])
    candidate = _result("latchmoe-eager", [[1, 2, 9], [4, 5]])

    comparison = compare_greedy_results(reference, candidate)

    assert comparison["match"] is False
    assert comparison["mismatched_requests"] == [0]
    with pytest.raises(CorrectnessMismatchError, match="request 0"):
        require_greedy_match(reference, candidate)


def test_greedy_comparison_rejects_different_manifest():
    reference = _result("uva", [[1], [2]])
    candidate = _result("latchmoe-eager", [[1], [2]])
    candidate["manifest_sha256"] = "b" * 64

    with pytest.raises(CorrectnessMismatchError, match="manifest"):
        require_greedy_match(reference, candidate)


def test_greedy_comparison_rejects_wrong_roles_and_sampling():
    reference = _result("uva", [[1], [2]])
    candidate = _result("latchmoe-eager", [[1], [2]])
    candidate["backend"] = "uva"
    candidate["sampling"] = {"temperature": 0.0, "max_tokens": 8, "seed": 0}

    with pytest.raises(CorrectnessMismatchError, match="candidate backend.*sampling"):
        require_greedy_match(reference, candidate)


@pytest.mark.parametrize(
    ("field", "invalid_value"),
    [
        ("mode", "latchmoe-eager"),
        ("backend", "latchmoe"),
        ("enforce_eager", False),
        ("expected_wave_event", True),
    ],
)
def test_greedy_comparison_rejects_each_invalid_uva_policy_field(
    field: str, invalid_value: object
):
    reference = _result("uva", [[1], [2]])
    candidate = _result("latchmoe-eager", [[1], [2]])
    reference[field] = invalid_value

    with pytest.raises(CorrectnessMismatchError, match="reference"):
        require_greedy_match(reference, candidate)


def test_greedy_comparison_rejects_inconsistent_graph_policy():
    reference = _result("uva", [[1], [2]])
    candidate = _result("latchmoe-piecewise", [[1], [2]])
    candidate["enforce_eager"] = True

    with pytest.raises(CorrectnessMismatchError, match="graph policy"):
        require_greedy_match(reference, candidate)

    reference["expected_wave_event"] = True
    candidate["enforce_eager"] = False
    with pytest.raises(CorrectnessMismatchError, match="UVA graph policy"):
        require_greedy_match(reference, candidate)


def test_greedy_result_files_can_be_compared(tmp_path: Path):
    reference_path = tmp_path / "uva.json"
    candidate_path = tmp_path / "latchmoe.json"
    reference_path.write_text(json.dumps(_result("uva", [[1], [2]])))
    candidate_path.write_text(json.dumps(_result("latchmoe-eager", [[1], [2]])))

    from vllm_latchmoe_cuda.correctness import require_greedy_match_files

    comparison = require_greedy_match_files(reference_path, candidate_path)

    assert comparison["match"] is True


def test_correctness_runner_exposes_help():
    completed = subprocess.run(
        [sys.executable, str(ROOT / "scripts/run_correctness.py"), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "--artifact-dir" in completed.stdout
    assert "--reference-json" in completed.stdout
    assert "--final-repetitions" not in completed.stdout


def test_correctness_runner_records_invalid_manifest_failure(tmp_path: Path):
    manifest_path = tmp_path / "invalid-manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    artifact_dir = tmp_path / "failed-run"

    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts/run_correctness.py"),
            "--mode",
            "uva",
            "--kind",
            "smoke",
            "--artifact-dir",
            str(artifact_dir),
            "--manifest",
            str(manifest_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode != 0
    failure = json.loads((artifact_dir / "failure.json").read_text())
    run_manifest = json.loads((artifact_dir / "run_manifest.json").read_text())
    assert failure["type"] == "ManifestValidationError"
    assert run_manifest["status"] == "failed"
    assert (artifact_dir / "SHA256SUMS").is_file()
