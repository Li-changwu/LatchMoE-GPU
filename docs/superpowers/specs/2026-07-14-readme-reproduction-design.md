# README Reproduction Runbook Design

## Goal

Replace the current untracked README draft with a command-complete runbook that
lets an operator reproduce the validated CUDA LatchMoE implementation on an
empty server while preserving the frozen vLLM/model/manifest contract and the
project's evidence boundaries.

## Chosen Approach

Use one staged README whose commands progress from cheap environment checks to
real-model validation:

1. state supported scope and unverified scope;
2. reproduce and verify the software/hardware environment;
3. install the out-of-tree plugin without changing vLLM dependencies;
4. place and validate the exact Qwen checkpoint and frozen manifest;
5. run unit, integration, and CUDA regression tests;
6. run all 12 real Qwen layer comparisons;
7. run synthetic eager and PIECEWISE graph checks;
8. run UVA first, then each LatchMoE full-model correctness mode against the
   same UVA result;
9. inspect artifact manifests, logs, JSON, and SHA-256 inventories;
10. document failure diagnosis and the remaining performance-work boundary.

This is preferred over a short quick-start because a quick-start would blur
synthetic, layer-level, and end-to-end evidence. It is preferred over adding an
automation script because the repository does not yet expose a controlled
ShareGPT performance driver or CLI orchestration for repeated measurement and
final promotion.

## Sources of Truth

The README will derive commands and claims only from:

- `pyproject.toml` for package metadata and plugin entry point;
- `benchmark/manifests/offload_manifest.json` for model identity, vLLM 0.19.1,
  BF16, TP=1, 32 slots, selected layers, and canonical manifest SHA-256;
- `benchmark/scripts/generate_offload_manifest.py` for model-path-aware
  manifest generation and deterministic checking;
- `scripts/run_correctness.py` and `scripts/run_graph_checks.py` for exact CLI
  arguments and artifact behavior;
- `tests/real/` for opt-in environment variables and comparison flow;
- `docs/gpu_port/PROGRESS.md` for already observed results and known blockers;
- a fresh read of installed package, CUDA, driver, GPU, and host-memory versions.

No README statement may turn a pytest duration, smoke run, warmup, or single
measurement into a performance result.

## README Structure

### Status and Scope

Describe the three implemented stages and the controlled UVA adapter. State
that real layer correctness is verified on the source server, while full-model
greedy equivalence and UVA-versus-LatchMoE performance still require the empty
server. State that no ShareGPT benchmark command exists in the repository yet.

### Exact Environment

Record the observed source environment:

- NVIDIA RTX A6000, 46068 MiB, driver 580.159.03;
- host RAM 503 GiB;
- Python 3.13.12;
- PyTorch 2.10.0+cu128 with CUDA runtime 12.8;
- vLLM 0.19.1;
- Triton 3.6.0;
- transformers 5.10.2;
- safetensors 0.7.0;
- pytest 9.0.2 and Ruff 0.15.21.

The README will distinguish "reference environment" from minimum compatibility.
It will instruct the operator to fail if vLLM is not exactly 0.19.1.

### Checkout and Installation

Use branch `cuda-latchmoe` and record reference implementation commit
`205caead34b5c27f24ae32fdbdfcd5e18936014a`. Install the reference CUDA 12.8
torch wheel, vLLM, and exact broad-dependency versions before installing the
plugin with `python -m pip install -e . --no-deps`, so editable installation
cannot silently replace torch or vLLM. Include an executable version-print and
entry-point discovery check.

### Model and Manifest

For exact reproduction, require the model at
`/root/models/Qwen3-30B-A3B-Instruct-2507` with revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`. Include a fixed-revision
`snapshot_download` path for empty servers, then run the frozen manifest check,
model-file validation API, and referenced-shard existence check before CUDA
tests.

If the operator must use another absolute model path, explain that regenerating
the manifest changes its canonical hash. Every UVA and LatchMoE run in that
experiment must then use the same regenerated manifest; it is a new experiment
contract, not byte-identical reproduction of the original manifest.

### Staged Verification

Each stage will have an exact command, expected success condition, artifact
location, and evidence boundary:

- environment/manifest checks: configuration evidence only;
- normal pytest: implementation regression evidence;
- real-layer pytest: layer-level Qwen numerical evidence;
- graph runner: synthetic-runtime smoke evidence only;
- full-model correctness runner: token-level greedy evidence when UVA and a
  candidate result match under the complete contract.

Artifact directory names will be unique and timestamped or server-labeled
because `ArtifactRun.create()` rejects existing directories. UVA must run first.
Each candidate must name UVA's `correctness.json` through `--reference-json`.

### Artifact Integrity

Explain `run_manifest.json`, `correctness.json` or `graph_checks.json`, profile
JSONL, stdout/stderr, `artifact_inventory.json`, and `SHA256SUMS`. Require
`status=completed`, `exit_code=0`, and `final_result=false` for smoke artifacts.
Explain that `kind=measurement` on one run is not a final result and that final
promotion requires at least three validated child measurements.

### Troubleshooting

Include the vLLM startup-memory formula
`total_memory * gpu_memory_utilization`, explain why 14 GiB offload has not run
when that gate is evaluated, and require an empty GPU rather than recommending
an artificially tiny utilization. Include failures for wrong vLLM version,
stale manifest, missing model hashes, existing artifact directories, missing
wave events, and token mismatch.

### Performance Boundary

State that the current repository does not yet expose the controlled ShareGPT
workload matrix, repeated benchmark CLI orchestration, or final statistical
comparison. The artifact library enforces final-promotion integrity, but the
README commands validate implementation and full-model correctness only; they
do not establish a UVA performance improvement.

## Verification of the README

Before committing:

1. run both runner `--help` commands and compare every documented flag;
2. run manifest determinism and model validation;
3. run the complete pytest suite;
4. run Ruff and formatting checks;
5. scan README commands for nonexistent files and bare runner invocations;
6. verify links and the GitHub branch/commit identifiers;
7. inspect the final diff and preserve the user's unrelated changes.

Commands requiring an empty full GPU will be documented but not claimed as
rerun on the occupied source server. Existing raw artifacts remain the only
evidence for previously observed source-server outcomes.
