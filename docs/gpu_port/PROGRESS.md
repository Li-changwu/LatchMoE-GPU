# GPU Port Progress

## Project Contract

- Target runtime: vLLM 0.19.1 only.
- vLLM tag commit: `b1388b1fbf5aaef47937fabe98931211684666a6`.
- Formal local model: `/home/lcw/model`, Qwen3-30B-A3B, revision
  `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Dtype / parallelism: BF16 / TP=1.
- Formal local offload manifest: layers 0-11, 32 slots, 13.5 GiB manifest
  expert weights plus residual stock UVA for byte parity.
- Baseline: official `vllm.model_executor.offloader.uva.UVAOffloader` with the
  same actual offload byte count.
- Formal workload: 50 ShareGPT requests, 128 forced output tokens,
  concurrency 8, two warmups and three measurements per backend.
- The original `/root/models/Qwen3-30B-A3B-Instruct-2507` contract below is
  retained as implementation history; it is not the 2026-07-15 measurement
  contract.

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

## 2026-07-15 - CUDA Device Planner and Formal ShareGPT Measurement

### Local Runtime and Port Completion

- Moved all active work onto the requested `cuda-latchmoe` branch without
  creating a new branch. The formal measurements bind to clean source commit
  `63a7ed49f4fa1b83d307bf7e28c291af9a31be0d`.
- Added support for the local `/home/lcw/model` Qwen3-30B-A3B checkpoint and
  revision `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39`.
- Selected layers 0-11 in
  `offload_manifest.qwen3-base-ad44.first12.local.json`. Each layer has all 128
  BF16 experts and 32 persistent CUDA slots.
- Kept direct checkpoint loading into the pinned HostStore, stable slot/map
  addresses, the protected LRU state machine, transfer stream/events, shared
  double stage banks, and capacity-bounded exact pair waves from the NPU design.
- Replaced routed-pair CPU planning with a CUDA device planner. Only the active
  expert set, at most 128 integer IDs, crosses to CPU. Pair offsets, logical and
  physical IDs, and routing weights stay on CUDA.
- Replaced per-wave scatter with one layer-level `index_add_`. Formal profile
  events report `pair_planner_mode=cuda_device` and
  `scatter_mode=layer_index_add`.
- Added the repeated ShareGPT server runner and summary comparator. The runner
  uses vLLM 0.19.1 `bench serve` metrics, enforces final-result promotion
  rules, validates equal actual offload bytes, and bypasses proxy variables for
  localhost traffic.

### Correctness Evidence and Boundary

- Final full test result with the local manifest and real-layer gate:
  `150 passed, 1 skipped`. The only skip is the strict full-model greedy comparison that
  requires an external UVA reference artifact. All 12 local real-layer tests
  executed.
- A real vLLM Triton slot comparison reported maximum absolute error
  `0.00390625` and mean absolute error approximately `3.53e-4`; it passed the
  BF16 numerical tolerance.
- Checkpoint samples matched their safetensors values and the real Triton
  layer-level numerical tests passed.
- A strict full-model greedy run did not reproduce UVA token-for-token. The
  evidence is consistent with small BF16 differences accumulating across the
  12 modified layers and changing an argmax near a decision boundary, but it
  does not prove that this is the only cause. Therefore the supported claim is
  layer-level numerical agreement within BF16 tolerance, not strict E2E token
  equivalence or output-quality equivalence.

### Frozen ShareGPT Contract

- Dataset:
  `/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json`.
- Dataset revision: `192ab2185289094fc556ec8ce5ce1e8e587154ca`.
- Size: `672837942` bytes; 94,145 records, of which 92,886 contain at least two
  turns.
- SHA-256:
  `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4`.
- Workload contract SHA-256:
  `bcda4b8da72118cc86fec7dba75258697b492247f927c659a335afb4f10af65b`.
- Exactly 50 requests selected with seed 42; no oversampling; request rate
  `inf`; maximum concurrency 8.
- Every request forces 128 output tokens with temperature 0 and `ignore_eos`.
  Each repetition contains 14,278 input and 6,400 output tokens.
- Both servers use BF16, TP=1, eager execution, `max_num_seqs=8`,
  `max_model_len=2048`, `max_num_batched_tokens=2048`, a 256 MiB explicit KV
  cache request, and disabled prefix caching.
- Each backend starts one server, runs two warmup requests, then runs three
  independent measurements against that same server.
- The official UVA baseline and LatchMoE both actually offload
  `15,798,475,264` bytes. LatchMoE consists of `14,495,514,624` manifest bytes
  plus `1,302,960,640` residual stock-UVA bytes. Stock UVA selects parameters in
  its native order; the byte count, not the exact parameter set, is controlled.

### Raw Measurement Results

All repetitions completed 50/50 requests with zero failures. Durations were
270.444, 266.733, and 268.123 seconds for LatchMoE; the metric table retains the
unrounded values in each `measurement.json`.

| Backend / repetition | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms/token | TPOT p99 ms/token | output token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| UVA / 1 | 8482.381 | 70348.791 | 504.165 | 858.751 | 11.8830 |
| UVA / 2 | 7753.382 | 65598.260 | 517.315 | 863.793 | 11.9639 |
| UVA / 3 | 7912.819 | 67985.256 | 499.073 | 827.669 | 12.0545 |
| LatchMoE / 1 | 4698.312 | 37694.233 | 244.752 | 396.570 | 23.6648 |
| LatchMoE / 2 | 3523.201 | 33762.247 | 258.502 | 378.561 | 23.9940 |
| LatchMoE / 3 | 4541.042 | 35582.418 | 243.907 | 401.222 | 23.8696 |

The comparison takes the median across the three repetitions independently for
each metric:

| Metric | official UVA | LatchMoE eager | improvement |
| --- | ---: | ---: | ---: |
| TTFT p50 | 7912.819 ms | 4541.042 ms | 42.61% lower |
| TTFT p99 | 67985.256 ms | 35582.418 ms | 47.66% lower |
| TPOT p50 | 504.165 ms/token | 244.752 ms/token | 51.45% lower |
| TPOT p99 | 858.751 ms/token | 396.570 ms/token | 53.82% lower |
| Output throughput | 11.9639 token/s | 23.8696 token/s | 99.51% higher |

The LatchMoE profile contains 3,394 `exact_waves` events. Real requests use two
to four waves, and the largest recorded layer event contains 15,480 routed
pairs. This confirms the benchmark exercised the device planner rather than
only the <=32-expert fast path.

### Formal Artifacts and Integrity

- Official UVA: `artifacts/sharegpt-c8-uva-20260715/`.
- LatchMoE eager:
  `artifacts/sharegpt-c8-latchmoe-eager-20260715/`.
- Comparison: `artifacts/sharegpt-c8-comparison-20260715.json`.
- Both formal `run_manifest.json` files record `status=completed`,
  `exit_code=0`, `final_result=true`, clean source, and the same source commit.
- `sha256sum -c SHA256SUMS` passes in both formal artifact directories.
- The UVA telemetry identifies the official implementation as
  `vllm.model_executor.offloader.uva.UVAOffloader`; LatchMoE telemetry identifies
  `vllm_latchmoe_cuda.offloader.CudaSEWOffloader` and its residual official UVA.
- `artifacts/sharegpt-c1-uva-20260715/` preserves a 50/50 request failure caused
  by localhost SOCKS proxying. `artifacts/sharegpt-c1-uva-20260715-r2/`
  preserves the deliberately interrupted concurrency-1 attempt. Neither failed
  artifact was overwritten or promoted.

### Scope and Capacity Limits

- This is an eager-versus-eager result. It must not be presented as a
  full-model PIECEWISE CUDA Graph result.
- The LatchMoE server reached approximately 45.37 GiB allocated/used device
  memory during measurement, leaving approximately 117 MiB free. The run did
  not OOM, but the margin is too small to claim portable capacity safety across
  other drivers, allocator states, concurrency levels, or model revisions.
- Fixed 128-token `ignore_eos` output makes serving work comparable, but the
  experiment does not measure generated answer quality. Combined with the
  strict greedy mismatch above, performance and quality claims must remain
  separate.

## 2026-07-15: Full-Model PIECEWISE Graph-to-Graph Measurement

### Root Cause and Graph Integration

The previous full-model LatchMoE path was eager and therefore did not test the
main execution-graph claim. The graph implementation now keeps dynamic expert
staging outside capture while placing the real vLLM modular Triton MoE kernel
inside a captured segment:

- `latchmoe::stage_experts` and `latchmoe::finish_experts` are explicit vLLM
  PIECEWISE splitting operators.
- `latchmoe::fused_moe_compute` is an opaque custom operator inside the captured
  segment.
- Graph mode uses one shared 128-slot identity pool. Logical expert `e` always
  uses slot `e`, so weight, map, and graph-token addresses remain stable across
  capture and replay. The pool is 1,207,959,552 bytes (1.125 GiB), not one pool
  per layer.
- Compile caching is disabled for benchmark servers because the vLLM AOT key
  does not encode the Latch manifest and instance-local layer rewrites.
- `max_num_batched_tokens` is 512 for both backends. A 2048-token Latch graph
  compile exceeded the available A6000 memory.

A correctness investigation found that vLLM advances
`ForwardContext.moe_layer_index` inside its stock MoE custom operator. LatchMoE
bypassed that operator without advancing the index. The first non-offloaded MoE
layer therefore resolved to layer 0, and every later MoE layer was shifted.
`vllm_context.advance_moe_layer_index()` now performs the same ordered advance
in eager mode and in graph-external staging. After this fix, the Latch and UVA
PIECEWISE smoke prompts produced the same short deterministic output.

### Capture and Replay Evidence

Both formal profiles contain a capture and a later replay with
`runtime_mode=PIECEWISE`. Both servers pre-captured token sizes 1, 2, 4, and 8
during startup. Runtime graph metrics repeatedly report PIECEWISE replay for
decode batches. Prefill chunks such as 512 tokens report runtime mode `NONE`, so
the supported claim is decode/mixed-batch PIECEWISE graph-to-graph, not a CUDA
Graph for arbitrary prompt shapes.

### Frozen Contract and Results

- Model: `/home/lcw/model`, Qwen3-30B-A3B, BF16, TP=1.
- Workload: 50 ShareGPT requests, seed 42, 128 forced output tokens,
  concurrency 8, request rate `inf`, two warmups, and three repetitions.
- Manifest:
  `benchmark/manifests/offload_manifest.qwen3-base-ad44.first12.graph128.local.json`.
- Manifest hash:
  `3c5a0c8fad70caf6e1c697d2a2471b798fc93dc3158da6bfddb8f8575f51c5e5`.
- Workload contract hash:
  `e90a0c2fb04f968432984e64ced7ea56994c3dda53b00051b8192eaf73ec27bc`.
- Source-state hash shared by both formal runs:
  `855bc37b916756aa1cba66eec0ff1323454feadab9fc369d7d90f63cad272786`.
- Both implementations actually offload `15,798,475,264` bytes.

| Backend / repetition | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms/token | TPOT p99 ms/token | output token/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| UVA graph / 1 | 9942.701 | 90186.894 | 578.740 | 980.697 | 9.9876 |
| UVA graph / 2 | 9592.722 | 81067.206 | 594.548 | 927.957 | 10.1169 |
| UVA graph / 3 | 10274.072 | 83348.327 | 569.167 | 969.359 | 10.3244 |
| Latch graph / 1 | 4276.263 | 46739.238 | 376.666 | 690.005 | 16.1718 |
| Latch graph / 2 | 3846.913 | 52281.366 | 377.351 | 633.805 | 16.0259 |
| Latch graph / 3 | 4112.697 | 49741.911 | 381.442 | 587.834 | 16.2434 |

The median-across-repetitions comparison is:

| Metric | official UVA graph | LatchMoE graph | improvement |
| --- | ---: | ---: | ---: |
| TTFT p50 | 9942.701 ms | 4112.697 ms | 58.64% lower |
| TTFT p99 | 83348.327 ms | 49741.911 ms | 40.32% lower |
| TPOT p50 | 578.740 ms/token | 377.351 ms/token | 34.80% lower |
| TPOT p99 | 969.359 ms/token | 633.805 ms/token | 34.62% lower |
| Output throughput | 10.1169 token/s | 16.1718 token/s | 59.85% higher |

### Interpretation and Limits

Graph replay removes launch overhead for both implementations but does not make
UVA weights device resident. Stock UVA replays kernels whose stable weight
pointers still address pinned host memory; the GPU continues to fetch those
weights over PCIe. LatchMoE deduplicates active experts, stages them with bulk
H2D copies, and replays the MoE compute against HBM slots. The current graph
path has neither cross-layer prefetch overlap nor persistent cache hits because
all selected layers share one pool and invalidate it on owner changes. The
measured advantage is therefore primarily the explicit DMA/HBM data path, not
an unimplemented overlap claim.

The graph pool deliberately spends 1.125 GiB of device memory that stock UVA
does not reserve. Equal offload bytes do not imply equal free GPU memory. Stock
UVA also selects parameters in native order while LatchMoE selects manifest
experts plus residual stock UVA; exact parameter identities are not controlled.

Generated text is not strictly deterministic across repetitions, even within
one backend, and only 7-10 of 50 generated strings matched exactly between
backends in the corresponding repetitions. Fixed token counts and shapes make
this a serving-performance comparison, not proof of identical routing traces or
answer quality. Layer-level BF16 numerical checks and the short full-model smoke
remain the available correctness evidence.

### Formal Artifacts

- Official UVA graph:
  `artifacts/sharegpt-c8-uva-piecewise-20260715-r1/`.
- LatchMoE graph:
  `artifacts/sharegpt-c8-latchmoe-piecewise-20260715-r1/`.
- Comparison:
  `artifacts/sharegpt-c8-piecewise-comparison-20260715-r1.json`.

Both run manifests record `status=completed`, `exit_code=0`, and
`final_result=true`. `sha256sum -c SHA256SUMS` passes for both directories.
The final suite reports `161 passed, 1 skipped`; the skip is the opt-in full
Qwen greedy comparison that requires `LATCHMOE_RUN_E2E=1`.
