from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .manifest import OffloadManifest


class CorrectnessMismatchError(AssertionError):
    pass


def compare_exact_token_results(
    native: Mapping[str, Any],
    uva: Mapping[str, Any],
    latchmoe: Mapping[str, Any],
) -> dict[str, object]:
    """Compare three deterministic runs by prompt and generated token IDs.

    This is intentionally stricter than a BF16 numerical tolerance or a fixed
    output length: every request must have the same prompt tokenization and
    greedy token sequence across the native oracle, exact-selection UVA and
    LatchMoE implementations.
    """
    errors: list[str] = []
    documents = (("native", native), ("uva", uva), ("latchmoe", latchmoe))
    reference_prompts = native.get("prompts")
    reference_sampling = native.get("sampling")
    records: list[tuple[str, list[Mapping[str, Any]]]] = []
    for name, document in documents:
        outputs = document.get("outputs")
        if not isinstance(outputs, list) or not outputs:
            errors.append(f"{name} has no outputs")
            continue
        records.append((name, outputs))
    if not records:
        return {"schema_version": 1, "match": False, "errors": errors}
    reference_name, reference = records[0]
    mismatches: dict[str, list[int]] = {}
    for name, candidate in records[1:]:
        source = dict(documents)[name]
        if source.get("prompts") != reference_prompts:
            errors.append(f"{name} prompt contract differs")
        if source.get("sampling") != reference_sampling:
            errors.append(f"{name} sampling contract differs")
        if len(candidate) != len(reference):
            errors.append(f"{name} request count differs")
            continue
        bad: list[int] = []
        for index, (expected, actual) in enumerate(zip(reference, candidate)):
            if (
                expected.get("prompt") != actual.get("prompt")
                or expected.get("prompt_token_ids") != actual.get("prompt_token_ids")
                or expected.get("token_ids") != actual.get("token_ids")
            ):
                bad.append(index)
        if bad:
            mismatches[name] = bad
    return {
        "schema_version": 1,
        "reference": reference_name,
        "request_count": len(reference),
        "match": not errors and not mismatches,
        "errors": errors,
        "mismatched_requests": mismatches,
    }


def require_exact_token_gate(
    native: Mapping[str, Any],
    uva: Mapping[str, Any],
    latchmoe: Mapping[str, Any],
) -> dict[str, object]:
    comparison = compare_exact_token_results(native, uva, latchmoe)
    if comparison["match"]:
        return comparison
    details = list(comparison.get("errors", []))
    details.extend(
        f"{name} token IDs differ at requests {indices}"
        for name, indices in comparison.get("mismatched_requests", {}).items()
    )
    raise CorrectnessMismatchError("; ".join(details))


STOCK_UVA_CPU_OFFLOAD_GB = 14.0
CORRECTNESS_MAX_NUM_SEQS = 32
CORRECTNESS_MAX_CUDAGRAPH_CAPTURE_SIZE = 8


@dataclass(frozen=True)
class CorrectnessMode:
    name: str
    backend: str
    enforce_eager: bool
    expect_waves: bool
    overlap_enabled: bool = False


_MODES = {
    "native": CorrectnessMode("native", "native", True, False),
    "uva": CorrectnessMode("uva", "uva", True, False),
    "uva-exact": CorrectnessMode("uva-exact", "uva", True, False),
    "latchmoe-eager": CorrectnessMode("latchmoe-eager", "latchmoe", True, False),
    "latchmoe-async": CorrectnessMode(
        "latchmoe-async", "latchmoe", True, False, True
    ),
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
    reference_mode = reference.get("mode")
    if (
        reference_mode not in {"uva", "uva-exact"}
        or reference.get("backend") != "uva"
    ):
        contract_errors.append("reference must be the UVA backend")
    elif (
        reference.get("enforce_eager") is not True
        or reference.get("expected_wave_event") is not False
    ):
        contract_errors.append("reference UVA graph policy is inconsistent")
    if reference_mode == "uva-exact":
        if reference.get("uva_implementation") != "ManifestUVAOffloader":
            contract_errors.append("exact UVA reference must use ManifestUVAOffloader")
        if reference.get("manifest_controls_offload_selection") is not True:
            contract_errors.append("exact UVA reference must be manifest-controlled")
        for field, label in (
            ("plan_id", "plan ID"),
            ("selected_layer_ids", "selected layer IDs"),
            ("identity_lock_sha256", "identity lock"),
        ):
            if not reference.get(field) or reference.get(field) != candidate.get(
                field
            ):
                contract_errors.append(f"{label} differs")
    candidate_mode = candidate.get("mode")
    candidate_policies = {
        "latchmoe-eager": (True, False, False),
        "latchmoe-async": (True, False, True),
        "latchmoe-piecewise": (False, False, False),
        "latchmoe-waves": (True, True, False),
    }
    if (
        not isinstance(candidate_mode, str)
        or candidate_mode not in candidate_policies
        or candidate.get("backend") != "latchmoe"
    ):
        contract_errors.append("candidate backend must be LatchMoE")
    else:
        expected_eager, expected_waves, expected_overlap = candidate_policies[
            candidate_mode
        ]
        if (
            candidate.get("enforce_eager") is not expected_eager
            or candidate.get("expected_wave_event") is not expected_waves
            or candidate.get("overlap_enabled") is not expected_overlap
        ):
            contract_errors.append("candidate graph policy is inconsistent")
    if reference.get("manifest_sha256") != candidate.get("manifest_sha256"):
        contract_errors.append("manifest SHA-256 differs")
    contract_fields = (
        ("model_revision", "model revision"),
        ("dtype", "dtype"),
        ("tensor_parallel_size", "tensor parallel size"),
        ("max_num_seqs", "maximum sequence count"),
        ("max_model_len", "maximum model length"),
        ("kv_cache_memory_bytes", "KV cache reserve"),
        ("request_execution", "request execution policy"),
        ("sampling", "sampling configuration"),
        ("prompts", "prompt list"),
        ("diagnostic_residual_uva_max_bytes", "residual UVA budget"),
        ("backend_hbm_cache_bytes", "backend HBM cache"),
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
    max_num_seqs: int = CORRECTNESS_MAX_NUM_SEQS,
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
        "max_num_seqs": max_num_seqs,
        "seed": 0,
        "enable_prefix_caching": False,
        "disable_log_stats": True,
        "attention_config": {"backend": "FLASH_ATTN"},
    }
    if mode.name in {"uva", "uva-exact"}:
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
    max_num_seqs: int = CORRECTNESS_MAX_NUM_SEQS,
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
            max_num_seqs=max_num_seqs,
        )
    )
    sampling = sampling_params_cls(
        temperature=0.0,
        max_tokens=max_tokens,
        seed=0,
    )
    outputs: list[dict[str, object]] = []
    for prompt in prompts:
        generated = engine.generate([prompt], sampling, use_tqdm=False)
        if len(generated) != 1:
            raise RuntimeError("greedy run returned an unexpected request count")
        request_output = generated[0]
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
    raw_plan = os.getenv("VLLM_LATCHMOE_RESIDENCY_PLAN_JSON")
    raw_identity_lock = os.getenv("VLLM_LATCHMOE_IDENTITY_LOCK_JSON")
    plan_id = None
    selected_layer_ids = None
    backend_hbm_cache_bytes = 0
    if raw_plan:
        from .residency_plan import deserialize_residency_plan

        plan = deserialize_residency_plan(raw_plan)
        plan_id = plan.plan_id
        selected_layer_ids = list(plan.offloaded_layer_ids)
        backend_hbm_cache_bytes = int(plan.main_slot_cache_bytes)
    return {
        "schema_version": 1,
        "mode": mode.name,
        "backend": mode.backend,
        "enforce_eager": mode.enforce_eager,
        "expected_wave_event": mode.expect_waves,
        "overlap_enabled": mode.overlap_enabled,
        "manifest_sha256": manifest.manifest_sha256,
        "model_revision": manifest.model.revision,
        "dtype": manifest.dtype,
        "tensor_parallel_size": manifest.tensor_parallel_size,
        "max_num_seqs": max_num_seqs,
        "max_model_len": max_model_len,
        "kv_cache_memory_bytes": kv_cache_memory_bytes,
        "enable_prefix_caching": False,
        "request_execution": "sequential",
        "plan_id": plan_id,
        "selected_layer_ids": selected_layer_ids,
        "identity_lock_sha256": (
            hashlib.sha256(raw_identity_lock.encode("utf-8")).hexdigest()
            if raw_identity_lock
            else None
        ),
        "uva_implementation": (
            "ManifestUVAOffloader"
            if mode.name == "uva-exact"
            else ("vllm-stock" if mode.name == "uva" else None)
        ),
        "cpu_offload_gb": (
            STOCK_UVA_CPU_OFFLOAD_GB
            if mode.name in {"uva", "uva-exact"}
            else 0.0
        ),
        "manifest_controls_offload_selection": mode.name != "uva",
        "diagnostic_residual_uva_max_bytes": int(
            os.getenv("VLLM_LATCHMOE_DIAGNOSTIC_RESIDUAL_UVA_BYTES", "0")
        ),
        "backend_hbm_cache_bytes": backend_hbm_cache_bytes,
        "prompts": list(prompts),
        "sampling": {"temperature": 0.0, "max_tokens": max_tokens, "seed": 0},
        "outputs": outputs,
    }
