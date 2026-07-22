from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import OffloadManifest


class CorrectnessMismatchError(AssertionError):
    pass


STOCK_UVA_CPU_OFFLOAD_GB = 14.0
CORRECTNESS_MAX_NUM_SEQS = 32
CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE = 8


@dataclass(frozen=True)
class CorrectnessMode:
    name: str
    backend: str
    enforce_eager: bool
    expect_waves: bool


_MODES = {
    "uva": CorrectnessMode("uva", "uva", True, False),
    "latchmoe-eager": CorrectnessMode("latchmoe-eager", "latchmoe", True, False),
    "latchmoe-piecewise": CorrectnessMode(
        "latchmoe-piecewise", "latchmoe", False, False
    ),
    "latchmoe-waves": CorrectnessMode("latchmoe-waves", "latchmoe", True, True),
}


DEFAULT_PROMPTS = (
    "Explain why deterministic tests matter in one sentence.",
    "Write a short Python function that returns the square of an integer.",
    "用一句话说明可复现实验为什么重要。",
)


def resolve_correctness_mode(name: str) -> CorrectnessMode:
    try:
        return _MODES[name]
    except KeyError as exc:
        raise ValueError(
            f"unknown correctness mode {name!r}; expected one of {sorted(_MODES)}"
        ) from exc


def load_prompts(path: str | Path | None) -> list[str]:
    if path is None:
        return list(DEFAULT_PROMPTS)
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise ValueError("prompt file must contain a non-empty JSON string list")
    if any(not isinstance(prompt, str) or not prompt for prompt in payload):
        raise ValueError("every prompt must be a non-empty string")
    return payload


def _output_records(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise CorrectnessMismatchError("correctness result has no outputs list")
    return outputs


def compare_greedy_results(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    contract_errors: list[str] = []
    if reference.get("mode") != "uva" or reference.get("backend") != "uva":
        contract_errors.append("reference must be the UVA backend")
    elif (
        reference.get("enforce_eager") is not True
        or reference.get("expected_wave_event") is not False
    ):
        contract_errors.append("reference UVA graph policy is inconsistent")
    candidate_mode = candidate.get("mode")
    candidate_policies = {
        "latchmoe-eager": (True, False),
        "latchmoe-piecewise": (False, False),
        "latchmoe-waves": (True, True),
    }
    if (
        not isinstance(candidate_mode, str)
        or candidate_mode not in candidate_policies
        or candidate.get("backend") != "latchmoe"
    ):
        contract_errors.append("candidate backend must be LatchMoE")
    else:
        expected_eager, expected_waves = candidate_policies[candidate_mode]
        if (
            candidate.get("enforce_eager") is not expected_eager
            or candidate.get("expected_wave_event") is not expected_waves
        ):
            contract_errors.append("candidate graph policy is inconsistent")
    if reference.get("manifest_sha256") != candidate.get("manifest_sha256"):
        contract_errors.append("manifest SHA-256 differs")
    contract_fields = (
        ("model_revision", "model revision"),
        ("dtype", "dtype"),
        ("tensor_parallel_size", "tensor parallel size"),
        ("max_num_seqs", "maximum sequence count"),
        ("sampling", "sampling configuration"),
        ("prompts", "prompt list"),
    )
    for field, label in contract_fields:
        if reference.get(field) != candidate.get(field):
            contract_errors.append(f"{label} differs")
    reference_outputs = _output_records(reference)
    candidate_outputs = _output_records(candidate)
    if len(reference_outputs) != len(candidate_outputs):
        contract_errors.append(
            "output count differs: "
            f"reference={len(reference_outputs)}, candidate={len(candidate_outputs)}"
        )

    mismatched_requests: list[int] = []
    mismatched_text_requests: list[int] = []
    for index, (expected, actual) in enumerate(
        zip(reference_outputs, candidate_outputs)
    ):
        if expected.get("prompt") != actual.get("prompt"):
            contract_errors.append(f"request {index} prompt differs")
        if expected.get("prompt_token_ids") != actual.get("prompt_token_ids"):
            contract_errors.append(f"request {index} prompt token ids differ")
        if expected.get("token_ids") != actual.get("token_ids"):
            mismatched_requests.append(index)
        if expected.get("text") != actual.get("text"):
            mismatched_text_requests.append(index)

    return {
        "schema_version": 1,
        "reference_mode": reference.get("mode"),
        "candidate_mode": candidate.get("mode"),
        "request_count": min(len(reference_outputs), len(candidate_outputs)),
        "match": not contract_errors
        and not mismatched_requests
        and not mismatched_text_requests,
        "contract_errors": contract_errors,
        "mismatched_requests": mismatched_requests,
        "mismatched_text_requests": mismatched_text_requests,
    }


def require_greedy_match(
    reference: Mapping[str, Any], candidate: Mapping[str, Any]
) -> dict[str, object]:
    comparison = compare_greedy_results(reference, candidate)
    if comparison["match"]:
        return comparison
    details = list(comparison["contract_errors"])
    details.extend(
        f"request {index} token ids differ"
        for index in comparison["mismatched_requests"]
    )
    details.extend(
        f"request {index} decoded text differs"
        for index in comparison["mismatched_text_requests"]
    )
    raise CorrectnessMismatchError("; ".join(details))


def require_greedy_match_files(
    reference_path: str | Path, candidate_path: str | Path
) -> dict[str, object]:
    reference = json.loads(Path(reference_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    return require_greedy_match(reference, candidate)


def build_engine_kwargs(
    *,
    manifest: OffloadManifest,
    mode: CorrectnessMode,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int,
) -> dict[str, object]:
    compilation_config = None
    if not mode.enforce_eager:
        compilation_config = {
            "cudagraph_mode": "PIECEWISE",
            "custom_ops": ["+unquantized_fused_moe"],
            "max_cudagraph_capture_size": CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE,
        }
    kwargs: dict[str, object] = {
        "model": manifest.model.path,
        "revision": manifest.model.revision,
        "dtype": "bfloat16",
        "tensor_parallel_size": 1,
        "enforce_eager": mode.enforce_eager,
        "compilation_config": compilation_config,
        "gpu_memory_utilization": gpu_memory_utilization,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "max_model_len": max_model_len,
        "max_num_seqs": CORRECTNESS_MAX_NUM_SEQS,
        "seed": 0,
        "disable_log_stats": True,
    }
    if mode.name == "uva":
        kwargs["cpu_offload_gb"] = STOCK_UVA_CPU_OFFLOAD_GB
    return kwargs


def run_vllm_greedy(
    *,
    manifest: OffloadManifest,
    mode: CorrectnessMode,
    prompts: Sequence[str],
    max_tokens: int,
    max_model_len: int,
    gpu_memory_utilization: float,
    kv_cache_memory_bytes: int,
    llm_cls=None,
    sampling_params_cls=None,
) -> dict[str, object]:
    manifest.validate_model_files()
    manifest.validate_runtime(
        vllm_version="0.19.1",
        dtype="bfloat16",
        tensor_parallel_size=1,
    )
    if llm_cls is None or sampling_params_cls is None:
        from vllm import LLM, SamplingParams

        llm_cls = LLM if llm_cls is None else llm_cls
        sampling_params_cls = (
            SamplingParams if sampling_params_cls is None else sampling_params_cls
        )
    engine = llm_cls(
        **build_engine_kwargs(
            manifest=manifest,
            mode=mode,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            kv_cache_memory_bytes=kv_cache_memory_bytes,
        )
    )
    sampling = sampling_params_cls(
        temperature=0.0,
        max_tokens=max_tokens,
        seed=0,
    )
    generated = engine.generate(list(prompts), sampling, use_tqdm=False)
    outputs: list[dict[str, object]] = []
    for prompt, request_output in zip(prompts, generated, strict=True):
        if len(request_output.outputs) != 1:
            raise RuntimeError("greedy run returned more than one sequence")
        sequence = request_output.outputs[0]
        outputs.append(
            {
                "prompt": prompt,
                "prompt_token_ids": list(request_output.prompt_token_ids),
                "token_ids": list(sequence.token_ids),
                "text": sequence.text,
            }
        )
    return {
        "schema_version": 1,
        "mode": mode.name,
        "backend": mode.backend,
        "enforce_eager": mode.enforce_eager,
        "expected_wave_event": mode.expect_waves,
        "manifest_sha256": manifest.manifest_sha256,
        "model_revision": manifest.model.revision,
        "dtype": manifest.dtype,
        "tensor_parallel_size": manifest.tensor_parallel_size,
        "max_num_seqs": CORRECTNESS_MAX_NUM_SEQS,
        "uva_implementation": "vllm-stock" if mode.name == "uva" else None,
        "cpu_offload_gb": STOCK_UVA_CPU_OFFLOAD_GB if mode.name == "uva" else 0.0,
        "manifest_controls_offload_selection": mode.name != "uva",
        "prompts": list(prompts),
        "sampling": {"temperature": 0.0, "max_tokens": max_tokens, "seed": 0},
        "outputs": outputs,
    }
