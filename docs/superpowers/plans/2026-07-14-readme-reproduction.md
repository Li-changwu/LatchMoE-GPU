# README Reproduction Runbook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the incomplete README draft with a verified new-server
runbook for the current CUDA LatchMoE implementation.

**Architecture:** Keep README as the single operator entry point. Order checks
from immutable environment/model contracts through synthetic, layer-level, and
full-model correctness, with artifact integrity and performance boundaries
stated at every stage.

**Tech Stack:** Markdown, Git, Python 3.13, PyTorch 2.10, CUDA 12.8, vLLM
0.19.1, pytest, JSON/SHA-256 artifacts.

---

### Task 1: Rewrite README as a staged reproduction runbook

**Files:**
- Modify: `README.md`

- [x] **Step 1: Replace unsupported and incomplete instructions**

Remove the nonexistent `requirements.txt` reference and bare invocations of
`run_graph_checks.py` and `run_correctness.py`. Preserve the project purpose,
but explicitly identify implementation commit
`205caead34b5c27f24ae32fdbdfcd5e18936014a` and branch `cuda-latchmoe`.

- [x] **Step 2: Record the reference environment and executable self-check**

Document the observed package versions and include one Python block that
asserts vLLM 0.19.1, prints torch/CUDA/package versions, checks CUDA
availability, and verifies the `latchmoe_cuda` entry point.

- [x] **Step 3: Document model and manifest validation**

Use the exact model path and revision. Include:

```bash
python benchmark/scripts/generate_offload_manifest.py --check
python -c 'from pathlib import Path; from vllm_latchmoe_cuda.manifest import OffloadManifest; m = OffloadManifest.load(Path("benchmark/manifests/offload_manifest.json")); m.validate_model_files(); print(m.manifest_sha256)'
```

Explain how a different absolute model path creates a new manifest contract.

- [x] **Step 4: Document all verification tiers with exact arguments**

Include normal pytest, opt-in 12-layer Qwen pytest, both graph runner modes,
then UVA and all three LatchMoE full-model modes. Use unique artifact paths,
`--kind smoke`, max model length 512, 16 greedy output tokens,
`--gpu-memory-utilization 0.98`, and a 256 MiB KV cache. Each candidate names
the UVA `correctness.json` via `--reference-json`.

- [x] **Step 5: Document evidence and performance boundaries**

Describe run manifests, raw results, logs, profile events, inventories, and
SHA-256 verification. State that synthetic smoke, warmup, pytest duration, and
single measurement runs are not final performance results, and that ShareGPT
benchmark orchestration is not yet exposed.

### Task 2: Verify the README against the actual repository

**Files:**
- Verify: `README.md`
- Verify: `scripts/run_correctness.py`
- Verify: `scripts/run_graph_checks.py`
- Verify: `benchmark/scripts/generate_offload_manifest.py`

- [x] **Step 1: Verify both documented runner interfaces**

Run:

```bash
python scripts/run_correctness.py --help
python scripts/run_graph_checks.py --help
```

Expected: exit 0 and every README flag appears in help output.

- [x] **Step 2: Verify manifest and model identity**

Run:

```bash
python benchmark/scripts/generate_offload_manifest.py --check
python -c 'from pathlib import Path; from vllm_latchmoe_cuda.manifest import OffloadManifest; m = OffloadManifest.load(Path("benchmark/manifests/offload_manifest.json")); m.validate_model_files(); print(m.manifest_sha256)'
```

Expected: exit 0 and manifest hash
`55fb0e810af54f463fb8f0844698e30946d12dbedb1b476cdf01ca9633a0ed0b`.

- [x] **Step 3: Run repository regression and static checks**

Run:

```bash
python -m pytest tests -q
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Expected on the source environment: 123 passed, 13 explicitly gated skips;
all other commands exit 0.

- [x] **Step 4: Scan documentation for known invalid patterns**

Run `rg` checks proving README does not reference `requirements.txt`, invoke a
runner without required arguments, label smoke as final, or claim an unmeasured
UVA performance improvement.

### Task 3: Commit and publish documentation

**Files:**
- Add: `README.md`
- Add: `docs/superpowers/plans/2026-07-14-readme-reproduction.md`
- Modify: `docs/superpowers/specs/2026-07-14-readme-reproduction-design.md`

- [x] **Step 1: Review the exact diff and staged paths**

Run `git diff -- README.md docs/superpowers/plans/2026-07-14-readme-reproduction.md`
and `git status --short`. Stage only the README, plan, and synchronized README
design spec.

- [ ] **Step 2: Commit**

```bash
git commit -m "docs: add new-server validation runbook"
```

- [ ] **Step 3: Push the existing branch**

```bash
git push origin cuda-latchmoe
```

Expected: fast-forward `origin/cuda-latchmoe`; never force-push.
