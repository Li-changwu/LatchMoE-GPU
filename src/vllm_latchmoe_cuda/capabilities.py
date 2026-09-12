"""Runtime capability descriptor and qualified support matrix."""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass
from typing import Any, Mapping

from .errors import CapabilityError

# vLLM 0.19.1 fused_moe.py hash observed by the ABI probe on the locked wheel.
# Editable checkouts must replace/extend this set with their seam-file hash.
SUPPORTED_VLLM_SOURCE_HASHES: frozenset[str] = frozenset(
    {"607c0a459306a71ff7d01445772494367f3924098739bbd3b4f43020738297d4"}
)


@dataclass(frozen=True)
class CapabilityDescriptor:
    model_family: str
    router_owner: str
    shared_expert_representation: str
    dtype: str
    tensor_parallel_size: int
    expert_parallel: bool
    quant_method: str
    kernel_mode: str
    output_abi: str
    combine_owner: str
    graph_mode: str
    vllm_source_sha256: str = "unknown"

    def to_jsonable(self) -> dict[str, object]:
        return {
            "model_family": self.model_family,
            "router_owner": self.router_owner,
            "shared_expert_representation": self.shared_expert_representation,
            "dtype": self.dtype,
            "tensor_parallel_size": self.tensor_parallel_size,
            "expert_parallel": self.expert_parallel,
            "quant_method": self.quant_method,
            "kernel_mode": self.kernel_mode,
            "output_abi": self.output_abi,
            "combine_owner": self.combine_owner,
            "graph_mode": self.graph_mode,
            "vllm_source_sha256": self.vllm_source_sha256,
        }


def source_sha256(module: Any) -> str:
    try:
        path = inspect.getfile(module)
        with open(path, "rb") as handle:
            return hashlib.sha256(handle.read()).hexdigest()
    except (OSError, TypeError):
        return "unknown"


def shared_expert_weight_bytes(shared: Any) -> int:
    if shared is None or not hasattr(shared, "parameters"):
        return 0
    seen: set[int] = set()
    total = 0
    for parameter in shared.parameters(recurse=True):
        identity = id(parameter)
        if identity not in seen:
            seen.add(identity)
            total += int(parameter.numel() * parameter.element_size())
    for buffer in shared.buffers(recurse=True):
        identity = id(buffer)
        if identity not in seen:
            seen.add(identity)
            total += int(buffer.numel() * buffer.element_size())
    return total


def _shared_representation(experts_module: Any, shared: Any, quant: Any) -> str:
    if shared is None:
        return "none"
    kernel = getattr(quant, "moe_kernel", None)
    if getattr(kernel, "shared_experts", None) is not None:
        return "fused"
    if bool(getattr(experts_module, "use_overlapped", False)):
        return "mix_placement"
    parameters = (
        tuple(shared.parameters(recurse=True))
        if hasattr(shared, "parameters")
        else ()
    )
    buffers = (
        tuple(shared.buffers(recurse=True)) if hasattr(shared, "buffers") else ()
    )
    if any(
        tensor.device.type != "cuda" for tensor in (*parameters, *buffers)
    ):
        return "external_host"
    return "external_resident"


def describe_capabilities(
    experts_module: Any,
    *,
    model_family: str = "qwen3_moe",
    dtype: str = "bfloat16",
    tensor_parallel_size: int = 1,
    expert_parallel: bool = False,
    graph_mode: str = "piecewise",
    source_module: Any | None = None,
) -> CapabilityDescriptor:
    router = getattr(experts_module, "router", None)
    quant = getattr(experts_module, "quant_method", None)
    shared = getattr(experts_module, "_shared_experts", None)
    kernel = getattr(quant, "moe_kernel", None) or getattr(quant, "kernel", None)
    return CapabilityDescriptor(
        model_family=model_family,
        router_owner=f"{type(router).__module__}.{type(router).__name__}",
        shared_expert_representation=_shared_representation(experts_module, shared, quant),
        dtype=dtype.removeprefix("torch."),
        tensor_parallel_size=int(tensor_parallel_size),
        expert_parallel=bool(expert_parallel),
        quant_method=f"{type(quant).__module__}.{type(quant).__name__}",
        kernel_mode=("monolithic" if bool(getattr(quant, "is_monolithic", False)) else "modular")
        if kernel is not None or quant is not None
        else "unknown",
        output_abi="tuple" if bool(getattr(experts_module, "returns_shared_experts", False)) else "tensor",
        combine_owner="vllm.native",
        graph_mode=graph_mode,
        vllm_source_sha256=source_sha256(source_module) if source_module is not None else "unknown",
    )


def validate_capabilities(
    descriptor: CapabilityDescriptor,
    *,
    require_native_combine: bool = True,
) -> None:
    failures: list[str] = []
    if descriptor.model_family not in {"qwen3_moe"}:
        failures.append("model family is not qualified")
    if descriptor.dtype != "bfloat16":
        failures.append("only bfloat16 is qualified")
    if descriptor.tensor_parallel_size != 1:
        failures.append("tensor parallel size must be 1")
    if descriptor.expert_parallel:
        failures.append("expert parallel is unsupported")
    if descriptor.shared_expert_representation not in {"none", "external_resident"}:
        failures.append("shared expert must be an external resident module")
    if descriptor.kernel_mode != "modular":
        failures.append("monolithic kernels are unsupported")
    if descriptor.graph_mode not in {"piecewise", "eager"}:
        failures.append("graph mode must be piecewise or eager")
    if require_native_combine and descriptor.combine_owner != "vllm.native":
        failures.append("native combine owner is unavailable")
    if (
        descriptor.vllm_source_sha256 != "unknown"
        and descriptor.vllm_source_sha256 not in SUPPORTED_VLLM_SOURCE_HASHES
    ):
        failures.append("vLLM seam source hash is not in the locked support matrix")
    if failures:
        raise CapabilityError("unsupported LatchMoE capability tuple: " + "; ".join(failures))
