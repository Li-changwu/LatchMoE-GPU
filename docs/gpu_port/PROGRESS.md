# GPU Port Progress

## Project Contract

- Target runtime: vLLM 0.19.1 only.
- vLLM tag commit: `b1388b1fbf5aaef47937fabe98931211684666a6`.
- Model: `/root/models/Qwen3-30B-A3B-Instruct-2507`.
- Model revision: `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.
- Dtype / parallelism: BF16 / TP=1.
- Main offload manifest: 12 layers, 32 slots, approximately 14 GiB expert
  offload target.
- Baseline: controlled vLLM CUDA-UVA reading the identical manifest.
- Final performance claim: reserved for repeated real experiments and user
  confirmation.

## 2026-07-13 - Read-Only Audit and Design

### Completed

- Located and verified the Qwen3 model and all 16 weight shards.
- Located vLLM 0.19.1 under `/opt/miniconda3` and locked it as the only target.
- Verified vLLM 0.19.1 exposes `BaseOffloader`, `make_layers().wrap_modules()`,
  `post_init()`, native CUDA-UVA, expert maps, and PIECEWISE CUDA Graph support.
- Audited portable NPU components: ExpertKey, HostStore, slot state machine,
  LRU policy, transfer batching, exact pair-wave planner, and profile events.
- Confirmed `/root/LatchMoE` was initially empty and initialized a local Git
  repository for the standalone plugin.
- Locked the shared manifest contract, selected layer ids, and 32-slot capacity.
- Approved the out-of-tree plugin design with a narrow, version-guarded offloader
  factory registration.

### Commands

```bash
git ls-remote https://github.com/Li-changwu/vllm-moe-offload-ascend.git HEAD
python -m pip metadata inspection for vllm, torch, triton, and transformers
rg / nl inspection of vLLM 0.19.1 offloader, Qwen3 MoE, FusedMoE, and graph code
GitHub API/raw inspection of the NPU portable runtime modules
nvidia-smi, nvidia-smi topo -m, lscpu, free -h, df -hT
```

### Artifacts

- Design specification:
  `docs/superpowers/specs/2026-07-13-latchmoe-cuda-design.md`
- No benchmark or profiler artifact was produced in this design phase.
- Existing smoke or result files elsewhere on the machine were not used as
  evidence for this project.

### Failures and Blockers

- The first local design commit failed because Git had no configured author
  identity. No content was lost. Resolution: configure `Codex
  <codex@localhost>` in this repository only and retry the commit.
- The NPU source repository was available remotely but was not present locally
  during the audit.
- ShareGPT was not present locally during the audit.
- The GPU had an external, container-invisible allocation at audit time. Runtime
  experiments will re-check device availability before execution.
- Capacity feasibility is intentionally deferred to the experiment phase. It
  does not change the approved 14 GiB implementation contract.

### Next Step

The TDD implementation plan was written and self-reviewed at
`docs/superpowers/plans/2026-07-13-latchmoe-cuda-implementation.md`. It maps the
three stages to nine tasks with explicit RED/GREEN commands and evidence gates.
Execute Stage 1 from failing tests, then continue through Stages 2 and 3.

## 2026-07-13 - Stage 1: GPU Eager MVP Core

### Completed

- Added the deterministic shared `offload_manifest.json` with canonical
  SHA-256 validation, exact model/vLLM hashes, 12 selected layers, and 32 slots.
- Added portable ExpertKey, slot state, protected LRU, exact-wave descriptors,
  counters, and JSONL writer.
- Added a single contiguous BF16 pinned HostStore. Selected Parameters are bound
  to HostStore views inside `make_layers().wrap_modules()` before checkpoint
  loading while preserving the original Parameter and weight loader objects.
- Added persistent CUDA slot Parameters and an in-place int32 logical-to-physical
  map. Synthetic tests verified stable addresses across staging operations.
- Added synchronous staging for active unions up to 32 and typed capacity/stale
  mapping failures.
- Added a small BF16 MoE numerical comparison against full expert weights.
- Added the version-guarded `vllm.general_plugins` registration and a controlled
  native-UVA adapter that selects the identical manifest parameters.
- Installed the package editable into `/opt/miniconda3` without dependencies.

### TDD Evidence

- Manifest/ExpertKey RED: collection failed because the package did not exist.
- Portable core RED: collection failed because slots/policy/waves/profile modules
  did not exist.
- HostStore/direct-load RED: collection failed because HostStore/offloader
  modules did not exist.
- Stage 1 CUDA RED: collection failed because the runner adapter did not exist.
- Plugin/UVA RED: collection failed because plugin/UVA modules did not exist.
- One GREEN attempt exposed an invalid `union=1` test fixture; the fixture was
  corrected to repeat the same expert without changing the active union.

### Commands

```bash
/opt/miniconda3/bin/python -m pip install -e . --no-deps
/opt/miniconda3/bin/python benchmark/scripts/generate_offload_manifest.py --check
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration tests/gpu -q
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration tests/gpu -q \
  --junitxml=artifacts/stage1/pytest.xml
```

### Artifacts

- `benchmark/manifests/offload_manifest.json`
- `artifacts/stage1/pytest.xml` - 40 tests, 40 passed, 0 failed/skipped in the
  recorded run.
- Implementation commits: `0fbae84`, `d9f481f`, `2f8fd43`, `cc6316b`, and
  `02fb7ea` on `feature/cuda-latchmoe`.

### Scope Boundary

- The recorded Stage 1 run is implementation/synthetic correctness evidence.
- Real Qwen per-layer comparison and full greedy E2E comparison remain pending
  under the final real-model gate.
- No throughput, latency, bandwidth, or memory-capacity claim is made from this
  stage.

### Next Step

Implement Stage 2 async transfer, lifecycle events, eager split boundary, and
PIECEWISE CUDA Graph checks from failing tests.
