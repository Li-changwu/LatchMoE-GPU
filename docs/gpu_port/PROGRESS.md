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
