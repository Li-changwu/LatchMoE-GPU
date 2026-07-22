# LatchMoE GPU Design Review - 2026-07-16

## Scope

This review compares the Ascend ACLGraph control-plane seam with the CUDA port,
then checks the CUDA-specific memory, transfer, graph, and validation paths on an
RTX A6000 with vLLM 0.19.1 and PyTorch 2.10.0+cu128.

The target remains Qwen3-30B-A3B BF16, TP=1, with layers 0-11 selected by the
offload manifest. The stock comparison is vLLM `UVAOffloader` with the same actual
offload bytes.

## Architecture Findings

### 1. The graph path was not a finite-memory offload path

The prior adapter rejected PIECEWISE unless `num_slots == num_experts`. Its
captured kernel used logical ids directly and ignored the persistent `log2phy`
buffer. This was address-stable, but required 128 Qwen expert slots and separated
the CUDA Graph result from the 32-slot capacity-constrained eager design.

The CUDA modular MoE kernel already accepts `global_num_experts` plus an
`expert_map`. The finite-slot graph path now passes the persistent `log2phy` map
and keeps its address stable across replay. Capture is allowed only when:

```text
num_slots >= min(num_experts, max_capture_size * top_k)
```

This makes every captured routed pair fit in one main-slot working set. Shapes
outside the capture range may overflow and are executed as exact waves by the
eager custom-op implementation.

### 2. The double buffer duplicated too much device memory

The first finite-slot implementation allocated one main pool and two equally
sized stage banks. A 64-slot configuration therefore allocated 192 expert slots,
which is 50% more weight memory than the old 128-slot identity graph. The first
full-model smoke failed while allocating the final 192 MiB tensor, preserving the
failure in `artifacts/20260716-finite64-piecewise-smoke-v1/`.

Main capacity and wave capacity are now separate. The default wave capacity is
`min(main_slots, 32)`. Stage bank 0 is a view of the main pool and only bank 1 is
an extra allocation. A CUDA event records the last compute reading the shared
bank; the next main-pool transfer stream waits on it before overwriting weights.

For Qwen BF16, one expert is 9 MiB:

| Layout | Expert slots | Device weight region |
| --- | ---: | ---: |
| Old identity graph | 128 | 1.125 GiB |
| Naive finite 64 + 2x64 stage | 192 | 1.6875 GiB |
| Current 64 main + one extra 32 bank | 96 | 0.84375 GiB |

### 3. Exact waves paid for an unnecessary D2D copy

The old vLLM adapter copied both tensors from a ready stage bank into the main
slot Parameters before every wave. For 32 Qwen experts this is a 288 MiB D2D copy
per wave, in addition to the required H2D transfer. A union-128 prefill paid about
1.125 GiB of extra D2D traffic.

The adapter now resolves vLLM's modular MoE kernel and calls it with the stage
bank tensors and physical ids directly. The compatibility callback remains for
non-modular test adapters, but the real Qwen Triton path does not copy through the
main slots. The same direct path is used by non-captured overflow from the graph
custom op.

### 4. The correctness runner did not enable the graph adapter

`latchmoe-piecewise` previously selected a vLLM PIECEWISE compilation config but
did not set `VLLM_LATCHMOE_GRAPH_MODE=piecewise`. It could therefore label a run
as LatchMoE PIECEWISE without installing the custom router/stage/MLP path.

The runner now sets the adapter environment, disables the compile cache, fixes
capture sizes at 1/2/4/8, enables the unquantized custom op, and requires both
PIECEWISE capture and replay telemetry before accepting the result.

### 5. Small routing sets paid for a GPU sort kernel

The staging seam used `torch.unique(topk_ids).cpu()` for every layer. At
concurrency 1, Qwen decode has only eight routed ids. On the A6000, launching the
GPU unique kernel and copying its result took about 58.8 us, while copying the
raw 64 bytes and deduplicating on CPU took about 12.4 us. The new adaptive helper
uses raw D2H for at most 512 routed ids and retains GPU unique for larger sets.

The persistent `log2phy` publication also allocated a fresh 512-byte pinned CPU
tensor per layer and step. Runtimes now reuse two event-guarded pinned map
buffers, growing the pool only if both copies remain in flight.

## CUDA Features Used

- Dedicated H2D stream with pinned contiguous host storage.
- Coalesced `cudaMemcpyAsync` calls for adjacent expert and slot runs.
- Persistent main slot, `log2phy`, and graph-token addresses.
- CUDA events for compute-to-transfer and shared-bank ownership dependencies.
- CUDA Graph replay for bounded routed working sets.
- Direct Triton modular MoE execution from HBM stage banks.
- Device tensor construction of pair offsets, physical ids, weights, and scatter
  indices; only the unique expert list crosses to the host.
- Adaptive raw-ID D2H for small decode routes and reusable pinned map buffers.

## Evidence

### Automated tests

- Full suite: 174 passed, 1 explicitly gated E2E test skipped.
- Real checkpoint coverage: 12 selected Qwen layers, both bounded active sets and
  full-union exact waves.
- New CUDA test captures the finite-slot modular kernel, changes slot contents and
  `log2phy`, then replays the same graph and matches the full-weight reference.
- Overflow test verifies the shared stage-bank path and exact numerical output.

### Full-model correctness

The 64-slot PIECEWISE smoke completed with one observed capture and one replay.
The profile also recorded exact waves for non-captured prefill. Official UVA and
LatchMoE produced identical token ids for all three deterministic prompts.

- `artifacts/20260716-finite64-piecewise-smoke-v3/`
- `artifacts/20260716-finite64-uva-smoke-v2/`

### Exploratory performance

Workload: 50 ShareGPT requests, fixed 32 output tokens, concurrency 1,
`max_num_seqs=1`, capture size 1, one repetition. LatchMoE used 32 main slots and
a 32-slot wave bank. Both modes completed 50/50 requests, used the same workload
and source-state hashes, and offloaded 15,798,475,264 bytes.

| Metric | UVA PIECEWISE | LatchMoE 64-slot PIECEWISE | Improvement |
| --- | ---: | ---: | ---: |
| TTFT p50 | 2189.12 ms | 968.12 ms | 55.78% lower |
| TTFT p99 | 6177.56 ms | 2523.32 ms | 59.15% lower |
| TPOT p50 | 144.79 ms/token | 115.42 ms/token | 20.28% lower |
| TPOT p99 | 145.07 ms/token | 115.77 ms/token | 20.20% lower |
| Output throughput | 4.6069 token/s | 6.8072 token/s | 47.76% higher |

These are exploratory single-repetition results, not a final benchmark claim.
They precede the final small-route and pinned-map host-control optimizations, so
they are a conservative serving baseline for the current implementation. The
small-route microbenchmark is verified separately; its end-to-end effect must be
measured in the final three-run benchmark.

- `artifacts/20260716-c1-slot32-latchmoe-piecewise-exploratory-v1/`
- `artifacts/20260716-c1-slot32-uva-piecewise-exploratory-v1/`

## Remaining Limitations

1. Every selected layer still performs a device-to-host routing synchronization.
   Small sets avoid the GPU unique kernel, but DMA source selection still needs
   host ids. Removing the sync requires a different mechanism such as
   GPU-initiated UVA copy kernels, CUDA graph conditionals, or a routing predictor;
   none is yet proven faster than copy-engine staging for 9 MiB experts.
2. All 12 layers share one main pool. This minimizes memory but eliminates
   cross-token layer residency and serializes ownership. A configurable pool
   count is the next practical memory/performance tradeoff to evaluate.
3. Exact-wave planning creates per-wave device tensors and uses Python scheduling.
   Pair assignment is on device, but graph-capturing a fixed maximum wave loop or
   using a persistent CUDA scheduler remains future work.
4. The implementation targets unquantized modular Qwen MoE. Quantized, monolithic,
   expert-parallel, shared-expert, and multi-GPU paths remain fail-closed or
   unvalidated.
5. Wave capacity defaults to 32 and has not been tuned. `16/32/64` should be swept
   under prefill-heavy, decode-heavy, and mixed workloads.
6. Final single-concurrency performance requires the frozen 50-request,
   128-output-token, three-run benchmark for both 32-slot LatchMoE and UVA from
   one source state. Higher concurrency should be evaluated only after that gate.
