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

## 2026-07-13 - Stage 2: Async Lifecycle and CUDA Graph Core

### Completed

- Added a dedicated CUDA transfer stream and transfer-ready events.
- Added deterministic coalescing of adjacent expert/slot copies into contiguous
  H2D runs.
- Enforced `EMPTY -> LOADING -> READY -> COMPUTING -> READY`; `LOADING` is not
  published in the map and `COMPUTING` is not an eviction candidate.
- Added compute-done events. The transfer stream waits on prior compute events
  before a slot becomes reusable, without a host synchronization in the normal
  adapter path.
- Added capture rejection for dynamic staging and in-place map mutation.
- Added Dynamo-disabled eager prepare/finish boundaries around the original
  vLLM router and quant kernel call.
- Added an instance-local FusedMoE forward adapter. It reuses vLLM 0.19.1
  `router.select_experts()` and `quant_method.apply()` rather than copying the
  Qwen model or Triton kernel.
- Added a vectorized stable-address compute path and verified real CUDA Graph
  capture/replay on the synthetic layout.
- Added stale mapping, early reuse, graph break, pointer stability, and bounded
  CUDA allocator-growth tests.

### TDD Evidence

- Async/graph RED: collection failed because `ExpertCopy`, async transfer, and
  capturable compute did not exist.
- Runner integration RED: collection failed because the instance adapter did
  not exist.
- GREEN: the Stage 2 focused suite passed 8/8 tests.
- Full regression artifact passed 48/48 tests.

### Commands

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/gpu/test_async_lifecycle.py \
  tests/gpu/test_graph_boundary.py \
  tests/gpu/test_memory_stability.py \
  tests/integration/test_runner_adapter.py -q
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration tests/gpu -q \
  --junitxml=artifacts/stage2/pytest.xml
```

### Artifacts

- `artifacts/stage2/pytest.xml` - 48 tests, 48 passed in the recorded run.
- Implementation commit: `3ab013e`.

### Scope Boundary

- The CUDA Graph test captured and replayed stable slot compute while staging
  remained outside capture.
- Full Qwen vLLM PIECEWISE graph counters and server-level eager ablation remain
  pending under the final real-model gate.
- No performance claim is made from the synthetic graph test.

### Next Step

Implement Stage 3 capacity-bounded exact pair waves, shared double stage banks,
transfer-aware issue order, and union-128 stress tests.

## 2026-07-13 - Stage 3: B2 Exact Pair Waves

### Completed

- Added exact routed-pair descriptors with unique flat offsets and explicit
  token/top-k/expert/weight fields.
- Added deterministic capacity-bounded waves with at most 32 experts per wave.
- Added shared layout-scoped double stage banks. Synthetic two-layer validation
  confirmed both layers reference the same two CUDA allocations.
- Added transfer-aware future-wave issue selection while preserving fixed
  compute/scatter order.
- Added per-bank compute-done events; the transfer stream waits before reusing a
  stage bank.
- Added pair microbatch construction and final token `index_add` accumulation.
- Added the vLLM-style overflow adapter path. Each wave copies from a stable
  stage bank into the stable main slot Parameters, updates the stable map in
  place, and invokes the original `quant_method.apply()` with top-k=1 pairs.
- Added prefill union-128, mixed decode/prefill, and union 120/127/128 stress
  coverage.
- GPU numerical tests compared exact waves with a full 128-expert small-dimension
  reference for both prefill and mixed workloads.

### TDD and Failure Evidence

- Exact-wave RED: collection failed because the overflow executor did not exist.
- Overflow-adapter RED: the existing adapter raised
  `ActiveExpertCapacityError` for 4 experts with 2 slots.
- The first scheduler GREEN run exposed a duplicate issue sequence: a completed
  wave was removed from `issued` and later prefetched again. A completed-wave set
  was added; the issue-order test then passed.
- A subsequent comparison failure was limited to expected BF16 output versus an
  FP32 reference dtype; the assertion was corrected to compare FP32 views under
  the fixed BF16 tolerance.
- Final Stage 3 full regression artifact passed 60/60 tests.

### Commands

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/unit/test_wave_stress.py \
  tests/gpu/test_wave_numerics.py \
  tests/gpu/test_wave_double_buffer.py \
  tests/integration/test_runner_adapter.py -q
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration tests/gpu -q \
  --junitxml=artifacts/stage3/pytest.xml
```

### Artifacts

- `artifacts/stage3/pytest.xml` - 60 tests, 60 passed in the recorded run.
- Implementation commit: `4020736`.

### Scope Boundary

- Correctness-first host planning is implemented.
- A CUDA device planner is intentionally not implemented; it remains a later
  optimization only after real-model correctness and profiling.
- These correctness/stress runs are not performance measurements.

### Next Step

Run formatting/static checks, real Qwen per-layer comparisons, graph artifact
runners, and the real greedy E2E capacity/correctness gate. Preserve every
failure as an artifact and do not convert smoke results into final results.

## 2026-07-13 - Correctness and Artifact Gate (In Progress)

### Resume Audit

- Continued in the isolated worktree
  `/root/LatchMoE/.worktrees/cuda-latchmoe` on
  `feature/cuda-latchmoe`; no main-worktree files were changed.
- Verified the installed target remains vLLM 0.19.1 with PyTorch 2.10.0+cu128
  and one CUDA device.
- The RTX A6000 reported 46068 MiB total, 39079 MiB used, and 6407 MiB free.
  The external allocation is a current environment constraint, not an inferred
  LatchMoE memory result.
- Verified the local Qwen checkpoint path and weight-index file remain present.

### Commands and Results

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/unit/test_artifacts.py tests/real/test_qwen_layers.py -q
```

Result: 4 passed and 12 skipped. All 12 skips were the explicit
`LATCHMOE_RUN_REAL=1` gate for real Qwen layer comparisons; they are not
recorded as passed real-model cases.

### Current Work

- Complete the correctness and graph artifact runners.
- Replace the E2E sentinel with a real token-id artifact comparison.
- Run the smallest real layer gate before attempting wider real-model coverage.

### Graph Smoke Artifacts

```bash
/opt/miniconda3/bin/python scripts/run_graph_checks.py \
  --mode eager --kind smoke \
  --artifact-dir artifacts/graph-eager-smoke
/opt/miniconda3/bin/python scripts/run_graph_checks.py \
  --mode piecewise --kind smoke \
  --artifact-dir artifacts/graph-piecewise-smoke
```

Both commands exited 0. The artifacts are explicitly scoped as
`synthetic-runtime` smoke checks. Eager reported two Dynamo graphs, one graph
break, and zero captured segments. PIECEWISE reported two Dynamo graphs, one
graph break, and one captured CUDA Graph segment. Both reported stable slot/map
addresses, matching replay output, and zero allocated-byte growth across the
measured staging loop. The capture guard was checked by forcing
`is_current_stream_capturing=True`; this is not a full-model vLLM graph claim.

Artifacts:

- `artifacts/graph-eager-smoke/graph_checks.json`
- `artifacts/graph-eager-smoke/run_manifest.json`
- `artifacts/graph-eager-smoke/artifact_inventory.json`
- `artifacts/graph-piecewise-smoke/graph_checks.json`
- `artifacts/graph-piecewise-smoke/run_manifest.json`
- `artifacts/graph-piecewise-smoke/artifact_inventory.json`

### Failures

- The first layer-3 pytest invocation exited during setup because
  `artifacts/qwen-layer-3`, the parent of `--basetemp`, did not exist. No model
  weights or CUDA kernels were exercised. Resolution: create the artifact
  directory before rerunning the exact test node. This setup error is not a
  LatchMoE correctness result.

### Real Qwen Layer 3 Gate

```bash
LATCHMOE_RUN_REAL=1 /opt/miniconda3/bin/python -m pytest \
  'tests/real/test_qwen_layers.py::test_real_qwen_layer_matches_staged_eager_and_waves[3]' \
  -q --junitxml=artifacts/qwen-layer-3/pytest.xml \
  --basetemp=artifacts/qwen-layer-3/tmp
```

The corrected invocation passed one test in 18.61 seconds. The raw layer JSON
reported eager `max_abs=0.0`; the union-128 path executed four waves covering
128 pairs and reported `max_abs=0.005475044250488281` and
`mean_abs=0.00021655255113728344`. Both comparisons passed the fixed
`rtol=0.02`, `atol=0.02` threshold.

Artifacts:

- `artifacts/qwen-layer-3/pytest.xml`
- `artifacts/qwen-layer-3/tmp/test_real_qwen_layer_matches_s0/layer_3.json`

This is one real-layer correctness result, not an end-to-end or performance
result.

### All Manifest Layers - Real Qwen Correctness

```bash
LATCHMOE_RUN_REAL=1 /opt/miniconda3/bin/python -m pytest \
  tests/real/test_qwen_layers.py -q \
  --junitxml=artifacts/qwen-layers-all/pytest.xml \
  --basetemp=artifacts/qwen-layers-all/tmp
```

The command exited 0 with 13 passed in 172.91 seconds: one checkpoint-index
coverage test and 12 real layer comparisons. Structured parsing of the 12 raw
layer JSON files reported:

- layer ids: `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]`;
- one manifest hash:
  `55fb0e810af54f463fb8f0844698e30946d12dbedb1b476cdf01ca9633a0ed0b`;
- eager close: 12/12; maximum absolute error: `0.0`;
- exact-wave close: 12/12; maximum absolute error:
  `0.014952659606933594`; maximum layer mean absolute error:
  `0.00031606023549102247`;
- every layer recorded four waves and 128 routed pairs.

Artifacts:

- `artifacts/qwen-layers-all/pytest.xml`
- `artifacts/qwen-layers-all/tmp/**/layer_*.json` (12 raw files)
- `artifacts/qwen-layers-all/artifact_inventory.json`
- `artifacts/qwen-layers-all/SHA256SUMS`

The initial structured-summary attempt used `jq`, which is not installed. It
exited with `jq: command not found` and wrote no file. The successful read-only
summary used Python's standard JSON parser. No timing or throughput claim is
derived from these pytest durations.

### Full-Model Greedy Smoke Gate

The artifact runner was executed in four modes with the same frozen manifest,
BF16, TP=1, three fixed prompts, greedy temperature 0, 16 requested output
tokens, max model length 512, and a fixed 256 MiB KV cache request:

```bash
/opt/miniconda3/bin/python scripts/run_correctness.py \
  --mode uva --kind smoke --artifact-dir artifacts/qwen-uva-smoke \
  --max-tokens 16 --max-model-len 512 \
  --gpu-memory-utilization 0.98 --kv-cache-memory-bytes 268435456
/opt/miniconda3/bin/python scripts/run_correctness.py \
  --mode latchmoe-eager --kind smoke \
  --artifact-dir artifacts/qwen-eager-smoke \
  --max-tokens 16 --max-model-len 512 \
  --gpu-memory-utilization 0.98 --kv-cache-memory-bytes 268435456
/opt/miniconda3/bin/python scripts/run_correctness.py \
  --mode latchmoe-piecewise --kind smoke \
  --artifact-dir artifacts/qwen-piecewise-smoke \
  --max-tokens 16 --max-model-len 512 \
  --gpu-memory-utilization 0.98 --kv-cache-memory-bytes 268435456
/opt/miniconda3/bin/python scripts/run_correctness.py \
  --mode latchmoe-waves --kind smoke \
  --artifact-dir artifacts/qwen-waves-smoke \
  --max-tokens 16 --max-model-len 512 \
  --gpu-memory-utilization 0.98 --kv-cache-memory-bytes 268435456
```

All four commands exited 1 before weight loading. vLLM observed approximately
6.0 GiB free out of 44.42 GiB and rejected the 0.98 utilization request for
43.53 GiB. The device concurrently reported approximately 39 GiB in use by an
external allocation. This is a real environment memory blocker, not a UVA or
LatchMoE correctness result.

The PIECEWISE stdout confirms vLLM parsed `enforce_eager=False` and
`cudagraph_mode=PIECEWISE`; it does not prove full-model capture because engine
startup failed at the memory gate. No `correctness.json`, token comparison, or
`exact_waves` event was produced, so full-model greedy equivalence remains
unverified.

Each directory contains `run_manifest.json` with `status=failed` and
`final_result=false`, `contract.json`, environment, worker command, stdout,
stderr, `failure.json`, `artifact_inventory.json`, and `SHA256SUMS`:

- `artifacts/qwen-uva-smoke/`
- `artifacts/qwen-eager-smoke/`
- `artifacts/qwen-piecewise-smoke/`
- `artifacts/qwen-waves-smoke/`

## 2026-07-14 - Review Fixes and Final Candidate Gate

### Completed

- Fixed real vLLM `FusedMoE` registration when its pre-existing
  `_expert_map` attribute is `None`.
- Kept selected real expert Parameters on the pinned HostStore through vLLM
  post-load processing. The post-load path no longer materializes or retains a
  complete selected expert tensor on CUDA, and replaced Parameter references
  are collectible.
- Verified the real vLLM Triton `quant_method.apply()` path against the full
  reference for the selected Qwen layer layout.
- Cleared regular slot metadata before overflow execution so a
  regular-to-overflow-to-regular transition cannot publish a stale expert map.
- Kept the overflow executor outside Dynamo and rejected any capture-time
  dynamic staging.
- Published async expert maps from persistent pinned CPU backing with a
  nonblocking H2D copy and event lifetime guard. Stage 1 synchronous staging
  retains synchronous map publication.
- Replaced quadratic routed-pair duplicate checking with an O(n) set-based
  validation.
- Removed the controlled UVA adapter's redundant pinned backing allocation.
- Bound manifest layer id, shape, dtype, and stride fail-closed before direct
  loading.
- Extended the graph leak gate to enforce both allocated and reserved CUDA
  growth limits of 1 MiB.
- Tightened greedy result comparison across backend roles, model revision,
  dtype, TP, prompts, prompt token ids, sampling, and the complete eager/wave
  policy. The UVA reference must now be eager and must not expect a wave event.
- Closed final-result promotion bypasses. A final measurement now requires at
  least three distinct successful child measurement manifests with identical
  contract/source hashes, nonempty unique run ids, valid schema-1 result JSON,
  and matching result SHA-256 values.
- Preserved invalid-manifest and runtime failures as structured non-final
  artifacts.

### Review Findings and TDD Evidence

The first focused review found four Critical issues: a real `FusedMoE`
registration conflict, post-load full expert GPU materialization, stale slot
metadata after overflow, and a single-run final-repetition bypass. Those issues
were fixed before the candidate gates.

A subsequent focused review found two remaining contract issues: incomplete
child-measurement validation and incomplete UVA graph-policy validation. The
initial regression command produced exactly two expected failures. Additional
result schema/hash cases expanded this to seven expected failures before the
production fix.

The final independent review then reproduced two additional Important
integrity bypasses: symlinked child/result paths could escape the run root, and
JSON booleans/floats could pass integer equality checks. It also found two
Minor diagnostic/coverage gaps for malformed child manifests and the complete
UVA policy tuple. Twelve new cases failed before the fix. Physical paths now
use strict resolution and containment checks; JSON integers exclude booleans
and floats; result paths require strings; malformed children raise the typed
promotion error; and all four UVA policy fields are parameterized. The focused
suite then passed 41 tests. The same reviewer reran the bypass probes and
reported no remaining Critical, Important, or Minor findings with a `Yes`
readiness verdict.

```bash
/opt/miniconda3/bin/python -m pytest \
  tests/unit/test_correctness.py tests/unit/test_artifacts.py -q
```

### Real Qwen Layer Final Candidate

```bash
LATCHMOE_RUN_REAL=1 /opt/miniconda3/bin/python -m pytest \
  tests/real/test_qwen_layers.py -q \
  --junitxml=artifacts/qwen-layers-final-candidate/pytest.xml \
  --basetemp=artifacts/qwen-layers-final-candidate/tmp
```

The run exited 0 with 13 passed: one checkpoint-index coverage case and all 12
manifest layers. The raw summary reports 12/12 eager comparisons close, 12/12
exact-wave comparisons close, eager maximum absolute error `0.0`, exact-wave
maximum absolute error `0.014952659606933594`, four waves per layer, and 128
routed pairs per layer.

Artifacts:

- `artifacts/qwen-layers-final-candidate/pytest.xml`
- `artifacts/qwen-layers-final-candidate/summary.json`
- `artifacts/qwen-layers-final-candidate/tmp/**/layer_*.json`
- `artifacts/qwen-layers-final-candidate/artifact_inventory.json`
- `artifacts/qwen-layers-final-candidate/SHA256SUMS`

This is real layer-level correctness evidence. It is not full-model greedy,
CUDA Graph, latency, throughput, bandwidth, or capacity evidence.

### Synthetic Graph Final Candidates

```bash
/opt/miniconda3/bin/python scripts/run_graph_checks.py \
  --mode eager --kind smoke \
  --artifact-dir artifacts/graph-eager-final-candidate
/opt/miniconda3/bin/python scripts/run_graph_checks.py \
  --mode piecewise --kind smoke \
  --artifact-dir artifacts/graph-piecewise-final-candidate
```

Both commands exited 0 and explicitly record `scope=synthetic-runtime` and
`final_result=false`. Eager captured zero segments; PIECEWISE captured one.
Both report stable slot/map addresses, matching output, rejected capture-time
staging, and zero allocated/reserved growth under the 1 MiB gates.

Artifacts:

- `artifacts/graph-eager-final-candidate/`
- `artifacts/graph-piecewise-final-candidate/`

These smoke runs do not establish full-model PIECEWISE capture or performance.

### Current Full Regression

```bash
/opt/miniconda3/bin/python -m ruff format .
/opt/miniconda3/bin/python -m pytest tests -q \
  --junitxml=artifacts/final/pytest-final.xml
/opt/miniconda3/bin/python -m ruff check .
/opt/miniconda3/bin/python -m ruff format --check .
/opt/miniconda3/bin/python \
  benchmark/scripts/generate_offload_manifest.py --check
git diff --check
```

The regression command exited 0 with 123 passed, 13 skipped, and 18 dependency
deprecation warnings. The skips are the 12 explicit `LATCHMOE_RUN_REAL=1`
layer gates and one explicit `LATCHMOE_RUN_E2E=1` full greedy gate. The real
layer gates were run separately above. Ruff, format checking, manifest
determinism, and `git diff --check` all exited 0.

Artifact:

- `artifacts/final/pytest-final.xml`
- SHA-256:
  `3fd9ee19376d1311da47a963e5eae1c29e3b2e027fc7aeba48d9a481cd6e9d3d`

### Remaining Blocker and Next Step

Full-model greedy equivalence, full vLLM PIECEWISE capture, memory feasibility,
and performance remain unverified. All four controlled full-model smoke modes
still stop at vLLM's startup memory gate because only approximately 6 GiB of
the RTX A6000 is free while an external allocation consumes approximately 39
GiB. No UVA-versus-LatchMoE performance claim can be made from the available
artifacts.

After the external allocation is released, rerun all four full-model greedy
modes with the frozen manifest, then execute repeated measurement runs for UVA
and each LatchMoE mode under the same workload matrix. Only those raw JSON/log
and profiler artifacts can support the final improve/regress/workload-specific
claim, which remains subject to user confirmation.
