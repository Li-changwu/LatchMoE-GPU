# LatchMoE CUDA Plugin Design

## 1. Scope

This project ports the portable SEW/LatchMoE mechanisms from
`Li-changwu/vllm-moe-offload-ascend` to CUDA without copying the Ascend model
patching layer. The only supported runtime is vLLM 0.19.1 at upstream tag commit
`b1388b1fbf5aaef47937fabe98931211684666a6`.

The deliverable is an out-of-tree `vllm.general_plugins` package with three
incremental runtime stages:

1. synchronous eager expert staging;
2. asynchronous staging, lifecycle protection, and PIECEWISE CUDA Graph;
3. capacity-bounded exact pair waves for expert unions larger than the slot bank.

The target model is the local BF16 Qwen3-30B-A3B-Instruct-2507 checkpoint at
revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`, with TP=1.

## 2. Locked Experiment Contract

Both LatchMoE and the controlled vLLM CUDA-UVA baseline read the same immutable
manifest file:

`benchmark/manifests/offload_manifest.json`

The manifest records:

- schema version and canonical JSON SHA-256;
- model path, Hugging Face revision, config hash, and weight-index hash;
- vLLM version and tag commit;
- BF16 dtype and TP=1;
- 32 slots per offloaded layer;
- layer ids `[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]`;
- the exact `w13_weight` and `w2_weight` parameter names, shapes, strides, byte
  sizes, and pinned-store offsets for every selected layer.

Runtime consumers reject a manifest whose canonical hash, model hashes, dtype,
TP size, layer set, parameter layout, or vLLM version does not match. They do
not silently derive a replacement configuration.

## 3. Integration Choice

The package registers one general plugin entry point. During worker startup the
registration function verifies vLLM 0.19.1 and installs a guarded wrapper around
the `GPUModelRunner` offloader factory symbol. The wrapper returns:

- `CudaSEWOffloader` when `VLLM_LATCHMOE_MODE=latchmoe`;
- `ManifestUVAOffloader` when `VLLM_LATCHMOE_MODE=uva`;
- the original vLLM factory for every other mode.

The registration is idempotent and fails closed on unsupported vLLM versions.
No Qwen model class, decoder layer, Ascend platform patch, or vLLM CLI parser is
copied. Disabling the plugin restores native behavior because the original
factory remains the only path when the mode variable is absent.

`ManifestUVAOffloader` preserves the native vLLM 0.19.1 UVA data path:
pinned CPU allocation plus `get_accelerator_view_from_cpu_tensor`. It changes
selection only, using exact manifest parameter names instead of a byte-budget
prefix. This gives both methods the same offloaded tensors.

## 4. Portable Core

The CUDA-neutral core contains:

- `ExpertKey(layer_id, expert_id)` with stable ordering and JSON encoding;
- manifest parsing, canonical hashing, and runtime validation;
- pinned host layout metadata and expert bundle views;
- `SlotState`: `EMPTY`, `LOADING`, `READY`, and `COMPUTING`;
- `ExpertSlotBank` with generation counters and state transition checks;
- layer-scoped LRU selection that never evicts `LOADING` or `COMPUTING` slots;
- exact wave planning and routed-pair descriptors;
- structured counters and JSONL profile events.

Portable code imports neither `torch.npu` nor CUDA-specific APIs. Logic adapted
from the Ascend repository retains provenance in file headers and is reduced to
the behavior used by this CUDA design.

## 5. Direct Pinned Loading

`CudaSEWOffloader.wrap_modules()` runs inside vLLM's native `make_layers()`
lifecycle. For a selected decoder layer it locates the FusedMoE `w13_weight` and
`w2_weight` parameters and binds their `.data` to typed views of contiguous
pinned CPU storage before the checkpoint loader receives the model.

The checkpoint weight loaders remain attached to the original `Parameter`
objects and write directly into pinned CPU views. The implementation records the
initial temporary allocation separately and asserts that no loaded target expert
tensor is materialized as a complete CUDA weight tensor.

After weight loading and vLLM weight post-processing, `post_init()` validates the
host layout and allocates persistent CUDA tensors. Parameter objects are then
bound once to stable slot tensor views; subsequent requests update tensor content
with `copy_` and map content with `copy_`, never replace the tensors.

## 6. Stage 1: Eager MVP

Each selected layer owns persistent BF16 CUDA slot tensors for 32 experts and a
persistent CUDA int32 `log2phy`/`expert_map` tensor for all 128 logical experts.
The slot tensors use the layout expected by the unquantized Triton FusedMoE
kernel.

An instance-local FusedMoE runner adapter obtains routed top-k ids before kernel
execution. It computes the active union, synchronously stages missing expert
weights, updates the map in place, and calls the original unquantized kernel with
precomputed top-k weights and ids. If the union exceeds 32, Stage 1 raises a
typed capacity error and writes a profile event.

Stage 1 does not implement asynchronous overlap or wave splitting.

## 7. Stage 2: Async Lifecycle and CUDA Graph

Each CUDA runtime owns a dedicated transfer stream and reusable transfer events.
Contiguous source/destination runs are coalesced before non-blocking H2D copies.
The compute stream waits on a transfer-ready event before using new mappings.

Allowed slot transitions are:

```text
EMPTY -> LOADING -> READY -> COMPUTING -> READY
READY -> LOADING
READY -> EMPTY
```

`LOADING` cannot become visible in `log2phy`. `COMPUTING` cannot be overwritten
or evicted. A compute-done event guards the transition back to `READY`. Every map
entry carries the slot generation expected by the planner; stale generations
fail before kernel launch.

The eager splitting boundary lies between top-k routing and the Triton FusedMoE
kernel. Dynamic planning, LRU decisions, map updates, event recording, and H2D
staging execute under a Dynamo-disabled eager function. Persistent slot tensors,
map tensors, and Triton compute remain address-stable.

LatchMoE's graph mode is vLLM PIECEWISE CUDA Graph. The eager splitting boundary
separates uncaptured dynamic staging from captured compute segments. Capture is
rejected if a transfer or map mutation is attempted while
`torch.cuda.is_current_stream_capturing()` is true. `--enforce-eager` remains a
first-class ablation using the same manifest and runtime.

## 8. Stage 3: B2 Exact Pair Waves

When the active expert union exceeds 32, the planner builds capacity-bounded
waves. It flattens the routing result into uniquely indexed `(token, topk_pos,
expert, weight)` pairs. Each pair is assigned to exactly one wave and every wave
contains at most 32 distinct experts.

For each wave the runtime:

1. prepares a pair microbatch in original pair order;
2. assigns wave experts to physical slots;
3. stages weights into one of two shared layout-scoped stage banks;
4. runs Triton FusedMoE with top-k=1 over the pair microbatch;
5. scatter-adds pair outputs to their original token rows.

The final output is valid only if the descriptor proves that the assigned pair
count equals `num_tokens * top_k`, no pair id is duplicated, and no pair is
missing. Shared-expert output, when present, is computed once per original token
batch rather than once per wave.

The two stage banks alternate. Transfer into a bank waits for its prior
compute-done event. Transfer-aware host planning may reorder issue time but never
changes the deterministic compute/scatter order. A CUDA device planner is outside
the required implementation; it is considered only after host-planner
correctness and profiling are complete.

## 9. Failure Handling and Invariants

The runtime fails closed for:

- manifest or vLLM version mismatch;
- non-BF16 or TP other than 1;
- unexpected weight shape, stride, or parameter name;
- unpinned target host storage when pinned memory is available;
- active expert ids outside `[0, 127]`;
- Stage 1/2 union larger than 32;
- illegal slot transition or eviction of `LOADING`/`COMPUTING`;
- stale slot generation or inconsistent expert mapping;
- slot/map data-pointer change after initialization;
- staging or mapping mutation during CUDA Graph capture;
- missing or duplicate pair assignments in Stage 3.

Errors include the layer, request/step id, expert ids, slot states, and manifest
hash where available. Failures are written to structured artifacts before the
process exits when the logger is initialized.

## 10. Verification Strategy

Verification is layered so a small test cannot be confused with an end-to-end
result:

- CPU unit tests: manifest canonicalization, ExpertKey, host offsets, state
  transitions, LRU guards, wave capacity, pair uniqueness, and statistics;
- synthetic CUDA tests: pinned direct load, slot layout, H2D batching, event
  ordering, map generation, address stability, capture rejection, and leak loops;
- real per-layer tests: compare each selected Qwen layer's staged eager and wave
  outputs with the full-weight reference at BF16 tolerances;
- end-to-end correctness: deterministic greedy prompts compare token ids and
  decoded output between full reference where feasible, controlled UVA, eager
  LatchMoE, and graph LatchMoE;
- graph checks: compilation counters, graph-break reason, captured segment count,
  stable pointers before/after replay, and absence of staging inside capture;
- stress checks: prefill, mixed prefill/decode, and unions approaching 128, with
  exact pair accounting and repeated memory snapshots.

Capacity probes and performance experiments are separate later activities. An
OOM or infeasible capacity result is retained as a negative artifact rather than
reclassified as a successful correctness run.

## 11. Artifacts and Progress

Commands write to a unique directory under `artifacts/` containing:

- `run_manifest.json` with environment and command;
- stdout/stderr logs;
- test or benchmark JSON;
- CUDA memory snapshots where requested;
- profiler JSONL;
- a SHA-256 inventory of produced artifacts.

Smoke, warmup, and single-run outputs are labeled in their run manifest and are
never promoted to final performance results. Each stage appends commands,
artifacts, failures, and next steps to `docs/gpu_port/PROGRESS.md`.
