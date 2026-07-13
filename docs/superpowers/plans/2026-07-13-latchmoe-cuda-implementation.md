# LatchMoE CUDA Plugin Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and verify all three LatchMoE CUDA stages as an out-of-tree vLLM 0.19.1 plugin using one frozen manifest shared with a controlled native-UVA adapter.

**Architecture:** A version-guarded general plugin selects either `CudaSEWOffloader` or `ManifestUVAOffloader` at vLLM's existing `make_layers().wrap_modules()` lifecycle. Portable manifest, slot, LRU, wave, and statistics modules are separated from CUDA host storage, transfer, runtime, runner-adapter, and graph-boundary modules. Dynamic staging remains eager while stable slot tensors and Triton FusedMoE compute are eligible for PIECEWISE CUDA Graph capture.

**Tech Stack:** Python 3.13, PyTorch 2.10, CUDA, Triton through vLLM 0.19.1, pytest, JSON/JSONL artifacts, safetensors metadata.

---

## File Map

- `pyproject.toml`: package metadata, pytest configuration, and `vllm.general_plugins` entry point.
- `src/vllm_latchmoe_cuda/plugin.py`: exact-version guard and narrow offloader factory registration.
- `src/vllm_latchmoe_cuda/errors.py`: typed manifest, capacity, lifecycle, mapping, graph, and pair-integrity errors.
- `src/vllm_latchmoe_cuda/manifest.py`: canonical schema, hash verification, and runtime validation.
- `src/vllm_latchmoe_cuda/core/expert_key.py`: portable expert identity.
- `src/vllm_latchmoe_cuda/core/slots.py`: slot state machine and generation checks.
- `src/vllm_latchmoe_cuda/core/policy.py`: protected LRU selection.
- `src/vllm_latchmoe_cuda/core/waves.py`: exact pair index, capacity waves, issue order, and scatter descriptors.
- `src/vllm_latchmoe_cuda/profile.py`: counters and structured JSONL event writer.
- `src/vllm_latchmoe_cuda/host_store.py`: contiguous pinned storage and direct parameter binding.
- `src/vllm_latchmoe_cuda/transfer.py`: synchronous/asynchronous contiguous H2D runs and events.
- `src/vllm_latchmoe_cuda/runtime.py`: persistent slot banks, maps, lifecycle, eager staging, and double stage buffers.
- `src/vllm_latchmoe_cuda/split_ops.py`: Dynamo-disabled eager staging boundary and capture rejection.
- `src/vllm_latchmoe_cuda/runner_adapter.py`: instance-local vLLM FusedMoE routing/kernel adapter.
- `src/vllm_latchmoe_cuda/offloader.py`: `CudaSEWOffloader` lifecycle integration.
- `src/vllm_latchmoe_cuda/uva.py`: exact-manifest adapter preserving native vLLM UVA views.
- `benchmark/scripts/generate_offload_manifest.py`: deterministic frozen-manifest generator.
- `benchmark/manifests/offload_manifest.json`: shared 12-layer/32-slot contract.
- `scripts/run_correctness.py`: labeled layer and greedy correctness runner.
- `scripts/run_graph_checks.py`: PIECEWISE/eager graph and address checks.
- `tests/`: CPU, CUDA, vLLM integration, real-layer, and E2E tests.

### Task 1: Package, Manifest, and Portable Identity

**Files:**
- Create: `pyproject.toml`
- Create: `src/vllm_latchmoe_cuda/__init__.py`
- Create: `src/vllm_latchmoe_cuda/errors.py`
- Create: `src/vllm_latchmoe_cuda/manifest.py`
- Create: `src/vllm_latchmoe_cuda/core/__init__.py`
- Create: `src/vllm_latchmoe_cuda/core/expert_key.py`
- Create: `benchmark/scripts/generate_offload_manifest.py`
- Create: `benchmark/manifests/offload_manifest.json`
- Test: `tests/unit/test_manifest.py`
- Test: `tests/unit/test_expert_key.py`

- [ ] **Step 1: Write failing schema and identity tests**

```python
def test_manifest_hash_rejects_mutation(frozen_manifest):
    loaded = OffloadManifest.load(frozen_manifest)
    payload = json.loads(frozen_manifest.read_text())
    payload["num_slots"] = 31
    frozen_manifest.write_text(json.dumps(payload))
    with pytest.raises(ManifestHashError):
        OffloadManifest.load(frozen_manifest)

def test_expert_key_orders_by_layer_then_expert():
    assert sorted([ExpertKey(7, 1), ExpertKey(3, 9)]) == [
        ExpertKey(3, 9), ExpertKey(7, 1)
    ]
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_manifest.py tests/unit/test_expert_key.py -q`

Expected: collection fails because `vllm_latchmoe_cuda` does not exist.

- [ ] **Step 3: Implement canonical schema and generator**

```python
@dataclass(frozen=True, order=True)
class ExpertKey:
    layer_id: int
    expert_id: int

@dataclass(frozen=True)
class OffloadManifest:
    schema_version: int
    manifest_sha256: str
    model: ModelIdentity
    vllm: VllmIdentity
    dtype: str
    tensor_parallel_size: int
    num_slots: int
    layers: tuple[LayerLayout, ...]

    @classmethod
    def load(cls, path: Path) -> "OffloadManifest":
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = payload.pop("manifest_sha256")
        actual = hashlib.sha256(canonical_json(payload)).hexdigest()
        if actual != expected:
            raise ManifestHashError(expected=expected, actual=actual)
        return cls.from_payload(payload, expected)
```

The generator emits exactly layers
`[3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43, 47]`, shapes
`w13=[128,1536,2048]`, `w2=[128,2048,768]`, BF16, TP=1, 32 slots, and the
audited model/vLLM hashes.

- [ ] **Step 4: Verify GREEN and generated-file determinism**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/unit/test_manifest.py tests/unit/test_expert_key.py -q
/opt/miniconda3/bin/python benchmark/scripts/generate_offload_manifest.py --check
```

Expected: all tests pass and `--check` reports the committed manifest is current.

### Task 2: Slot State, Protected LRU, Exact Waves, and Statistics

**Files:**
- Create: `src/vllm_latchmoe_cuda/core/slots.py`
- Create: `src/vllm_latchmoe_cuda/core/policy.py`
- Create: `src/vllm_latchmoe_cuda/core/waves.py`
- Create: `src/vllm_latchmoe_cuda/profile.py`
- Test: `tests/unit/test_slots.py`
- Test: `tests/unit/test_policy.py`
- Test: `tests/unit/test_waves.py`
- Test: `tests/unit/test_profile.py`

- [ ] **Step 1: Write failing lifecycle and exactness tests**

```python
def test_computing_slot_cannot_be_evicted():
    bank = ExpertSlotBank(2)
    slot = bank.begin_load(ExpertKey(3, 1), slot_id=0)
    bank.mark_ready(slot.slot_id, slot.generation)
    bank.begin_compute((slot.slot_id,))
    with pytest.raises(NoEvictableSlotError):
        LruPolicy().choose(bank, excluded={1})

def test_each_pair_is_assigned_exactly_once():
    ids = torch.tensor([[0, 40], [80, 127]])
    weights = torch.ones_like(ids, dtype=torch.float32)
    plan = plan_exact_waves(ids, weights, capacity=32)
    assert sorted(plan.all_pair_offsets()) == list(range(ids.numel()))
    assert len(set(plan.all_pair_offsets())) == ids.numel()
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_slots.py tests/unit/test_policy.py tests/unit/test_waves.py tests/unit/test_profile.py -q`

Expected: imports fail for missing portable modules.

- [ ] **Step 3: Implement guarded state and deterministic pair planning**

```python
class SlotState(str, Enum):
    EMPTY = "empty"
    LOADING = "loading"
    READY = "ready"
    COMPUTING = "computing"

@dataclass
class ExpertSlot:
    slot_id: int
    key: ExpertKey | None = None
    state: SlotState = SlotState.EMPTY
    generation: int = 0
    last_used: int = 0

@dataclass(frozen=True)
class PairDescriptor:
    pair_offset: int
    token_index: int
    topk_position: int
    expert_id: int
    weight: float
```

Implement legal-transition checks, generation validation, protected LRU, stable
wave ordering, transfer-aware issue ordering separated from compute order, pair
coverage validation, counters, and flushed JSONL writes.

- [ ] **Step 4: Verify GREEN**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit -q`

Expected: all CPU unit tests pass.

### Task 3: Pinned Host Store and Direct Weight Loading

**Files:**
- Create: `src/vllm_latchmoe_cuda/host_store.py`
- Create: `src/vllm_latchmoe_cuda/offloader.py`
- Test: `tests/unit/test_host_store.py`
- Test: `tests/integration/test_direct_load.py`

- [ ] **Step 1: Write failing direct-load tests**

```python
def test_bind_parameter_keeps_weight_loader_and_uses_pinned_view(cuda_available):
    module = TinyExperts(device="cuda")
    loader = module.w13_weight.weight_loader
    layout = tiny_manifest.layers[0].tensors[0]
    store = PinnedHostStore(tiny_manifest)
    store.bind_parameter(3, "w13_weight", module.w13_weight)
    assert module.w13_weight.weight_loader is loader
    assert module.w13_weight.device.type == "cpu"
    assert module.w13_weight.is_pinned()

def test_checkpoint_loader_writes_directly_to_host_view(tiny_bound_module):
    loaded = torch.arange(tiny_bound_module.w13_weight.numel()).reshape_as(
        tiny_bound_module.w13_weight
    )
    tiny_bound_module.w13_weight.weight_loader(
        tiny_bound_module.w13_weight, loaded, "w1", expert_id=0
    )
    assert torch.equal(
        tiny_bound_module.w13_weight[0, : loaded.shape[1] // 2],
        loaded[0, : loaded.shape[1] // 2],
    )
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_host_store.py tests/integration/test_direct_load.py -q`

Expected: fails because `PinnedHostStore` and `CudaSEWOffloader` are absent.

- [ ] **Step 3: Implement pinned slabs and lifecycle binding**

```python
class PinnedHostStore:
    def __init__(self, manifest: OffloadManifest, pin_memory: bool = True):
        self._slab = torch.empty(
            manifest.total_elements,
            dtype=torch.bfloat16,
            device="cpu",
            pin_memory=pin_memory,
        )

    def tensor_view(self, layer_id: int, name: str) -> torch.Tensor:
        layout = self.layout(layer_id, name)
        return self._slab[layout.offset : layout.offset + layout.numel].view(
            layout.shape
        )

    def bind_parameter(self, layer_id: int, name: str, param: nn.Parameter) -> None:
        view = self.tensor_view(layer_id, name)
        if tuple(param.shape) != tuple(view.shape):
            raise LayoutMismatchError(
                layer_id=layer_id,
                parameter=name,
                expected=tuple(view.shape),
                actual=tuple(param.shape),
            )
        param.data = view
```

`CudaSEWOffloader.wrap_modules()` maps generator index to the manifest layer id,
binds only exact `mlp.experts.w13_weight/w2_weight` parameters, and records the
layer/FusedMoE object for `post_init()`.

- [ ] **Step 4: Verify GREEN**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_host_store.py tests/integration/test_direct_load.py -q`

Expected: CPU fallback tests pass; CUDA-only tests pass when the device is free or
skip with an explicit device-unavailable reason.

### Task 4: Stage 1 Persistent Layout and Synchronous Eager Runtime

**Files:**
- Create: `src/vllm_latchmoe_cuda/transfer.py`
- Create: `src/vllm_latchmoe_cuda/runtime.py`
- Create: `src/vllm_latchmoe_cuda/runner_adapter.py`
- Test: `tests/gpu/test_slot_layout.py`
- Test: `tests/gpu/test_sync_staging.py`
- Test: `tests/gpu/test_eager_numerics.py`

- [ ] **Step 1: Write failing stable-layout and numeric tests**

```python
def test_slot_and_map_addresses_remain_stable(tiny_cuda_runtime):
    before = tiny_cuda_runtime.data_ptrs()
    tiny_cuda_runtime.stage_sync(3, active_experts=(1, 7, 9))
    tiny_cuda_runtime.stage_sync(3, active_experts=(2, 7, 10))
    assert tiny_cuda_runtime.data_ptrs() == before

def test_sync_staged_output_matches_full_reference(tiny_moe_case):
    expected = tiny_moe_case.full_reference()
    actual = tiny_moe_case.latchmoe_eager()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/gpu/test_slot_layout.py tests/gpu/test_sync_staging.py tests/gpu/test_eager_numerics.py -q`

Expected: fails because persistent slot runtime is absent.

- [ ] **Step 3: Implement synchronous staging and precomputed routing adapter**

```python
class CudaLayerRuntime:
    def stage_sync(self, active_experts: Sequence[int]) -> MappingSnapshot:
        active = dedupe_validate(active_experts, self.num_experts)
        if len(active) > self.num_slots:
            raise ActiveExpertCapacityError(
                layer_id=self.layer_id,
                active_count=len(active),
                slot_count=self.num_slots,
            )
        plan = self.plan_slots(active)
        for load in plan.loads:
            self.slot_w13[load.slot_id].copy_(self.host_w13[load.expert_id])
            self.slot_w2[load.slot_id].copy_(self.host_w2[load.expert_id])
        self.log2phy.copy_(plan.log2phy)
        return self.snapshot(plan)
```

The adapter uses vLLM 0.19.1 routing utilities to obtain top-k weights/ids, calls
the runtime, then invokes the existing unquantized quant method with stable slot
weights and `expert_map`.

- [ ] **Step 4: Verify GREEN**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit tests/integration tests/gpu/test_slot_layout.py tests/gpu/test_sync_staging.py tests/gpu/test_eager_numerics.py -q`

Expected: all available Stage 1 tests pass.

### Task 5: Plugin Registration and Controlled Native-UVA Adapter

**Files:**
- Create: `src/vllm_latchmoe_cuda/plugin.py`
- Create: `src/vllm_latchmoe_cuda/uva.py`
- Modify: `pyproject.toml`
- Test: `tests/integration/test_plugin.py`
- Test: `tests/integration/test_manifest_uva.py`

- [ ] **Step 1: Write failing registration and exact-selection tests**

```python
def test_plugin_rejects_non_0191(monkeypatch):
    monkeypatch.setattr(metadata, "version", lambda _: "0.19.0")
    with pytest.raises(UnsupportedVllmVersionError):
        register()

def test_uva_selects_only_manifest_parameters(tiny_decoder_layers):
    offloader = ManifestUVAOffloader(tiny_manifest)
    offloader.wrap_modules(iter(tiny_decoder_layers))
    assert offloader.offloaded_parameter_names == tiny_manifest.parameter_names
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/integration/test_plugin.py tests/integration/test_manifest_uva.py -q`

Expected: missing plugin/UVA classes.

- [ ] **Step 3: Implement idempotent factory wrapper and native UVA views**

```python
def register() -> None:
    if metadata.version("vllm") != "0.19.1":
        raise UnsupportedVllmVersionError(
            expected="0.19.1", actual=metadata.version("vllm")
        )
    runner = importlib.import_module("vllm.v1.worker.gpu_model_runner")
    if getattr(runner.create_offloader, "_latchmoe_wrapped", False):
        return
    native_factory = runner.create_offloader

    def create(config):
        mode = os.getenv("VLLM_LATCHMOE_MODE", "")
        if mode == "latchmoe":
            return CudaSEWOffloader(load_manifest_from_env())
        if mode == "uva":
            return ManifestUVAOffloader(load_manifest_from_env())
        return native_factory(config)
```

`ManifestUVAOffloader` mirrors vLLM's UVA allocation/view sequence but selects
only exact manifest layer parameters.

- [ ] **Step 4: Verify GREEN and entry-point discovery**

Run:

```bash
/opt/miniconda3/bin/python -m pip install -e . --no-deps
/opt/miniconda3/bin/python -m pytest tests/integration/test_plugin.py tests/integration/test_manifest_uva.py -q
/opt/miniconda3/bin/python -c 'from importlib.metadata import entry_points; print([e.name for e in entry_points(group="vllm.general_plugins") if "latchmoe" in e.name])'
```

Expected: tests pass and entry point prints `latchmoe_cuda`.

### Task 6: Stage 2 Async Lifecycle and PIECEWISE Graph Boundary

**Files:**
- Modify: `src/vllm_latchmoe_cuda/transfer.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Create: `src/vllm_latchmoe_cuda/split_ops.py`
- Modify: `src/vllm_latchmoe_cuda/runner_adapter.py`
- Test: `tests/gpu/test_async_lifecycle.py`
- Test: `tests/gpu/test_graph_boundary.py`
- Test: `tests/gpu/test_memory_stability.py`

- [ ] **Step 1: Write failing event, stale-map, capture, and leak tests**

```python
def test_computing_slot_waits_before_reuse(async_runtime):
    handle = async_runtime.begin_compute(layer_id=3, active_experts=(1, 2))
    with pytest.raises(NoEvictableSlotError):
        async_runtime.stage_async(3, active_experts=(3, 4), force_slots=(0, 1))
    async_runtime.end_compute(handle)
    async_runtime.wait_compute_done(handle)
    async_runtime.stage_async(3, active_experts=(3, 4), force_slots=(0, 1))

def test_dynamic_staging_rejects_capture(async_runtime, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    with pytest.raises(StagingDuringCaptureError):
        async_runtime.stage_async(3, active_experts=(1,))
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/gpu/test_async_lifecycle.py tests/gpu/test_graph_boundary.py tests/gpu/test_memory_stability.py -q`

Expected: async/capture behavior is missing.

- [ ] **Step 3: Implement stream/event ordering and eager split**

```python
@torch.compiler.disable
def eager_stage_and_map(runtime, layer_id, topk_ids):
    if torch.cuda.is_current_stream_capturing():
        raise StagingDuringCaptureError(layer_id)
    active = torch.unique(topk_ids).tolist()
    return runtime.stage_async(layer_id, active)
```

Implement coalesced contiguous copies on a dedicated stream, transfer-ready
events, compute-done events, generation snapshots, in-place map updates,
pointer guards, and a runner split that leaves Triton compute capturable.

- [ ] **Step 4: Verify GREEN in eager and PIECEWISE modes**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/gpu/test_async_lifecycle.py tests/gpu/test_graph_boundary.py tests/gpu/test_memory_stability.py -q
/opt/miniconda3/bin/python scripts/run_graph_checks.py --mode eager --artifact-dir artifacts/graph-eager-smoke
/opt/miniconda3/bin/python scripts/run_graph_checks.py --mode piecewise --artifact-dir artifacts/graph-piecewise-smoke
```

Expected: tests pass; smoke artifacts explicitly report graph breaks, captured
segments, stable pointers, and memory deltas.

### Task 7: Stage 3 B2 Exact Waves and Double Stage Buffers

**Files:**
- Modify: `src/vllm_latchmoe_cuda/core/waves.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Modify: `src/vllm_latchmoe_cuda/runner_adapter.py`
- Test: `tests/unit/test_wave_stress.py`
- Test: `tests/gpu/test_wave_numerics.py`
- Test: `tests/gpu/test_wave_double_buffer.py`

- [ ] **Step 1: Write failing union-128 and scatter exactness tests**

```python
@pytest.mark.parametrize("union", [33, 64, 127, 128])
def test_wave_plan_covers_large_union_once(union):
    ids, weights = routing_with_union(union, top_k=8)
    plan = plan_exact_waves(ids, weights, capacity=32)
    assert max(len(w.experts) for w in plan.waves) <= 32
    validate_pair_coverage(plan, expected_pairs=ids.numel())

def test_wave_output_matches_full_reference(union_128_case):
    expected = union_128_case.full_reference()
    actual = union_128_case.latchmoe_waves()
    torch.testing.assert_close(actual, expected, rtol=2e-2, atol=2e-2)
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_wave_stress.py tests/gpu/test_wave_numerics.py tests/gpu/test_wave_double_buffer.py -q`

Expected: large unions still raise Stage 1/2 capacity errors.

- [ ] **Step 3: Implement pair microbatches, alternating buffers, and scatter-add**

```python
def execute_exact_waves(self, x, topk_ids, topk_weights):
    plan = plan_exact_waves(topk_ids, topk_weights, self.num_slots)
    output = torch.zeros_like(x)
    for compute_index, wave in enumerate(plan.compute_waves):
        bank = self.stage_banks[compute_index % 2]
        self.stage_wave_async(bank, wave)
        micro = build_pair_microbatch(x, topk_ids, topk_weights, wave)
        pair_output = self.run_top1(bank, micro)
        output.index_add_(0, micro.token_indices, pair_output)
    validate_pair_coverage(plan, topk_ids.numel())
    return output
```

Preserve deterministic compute order, allow transfer-aware issue order only,
and guard both banks with transfer-ready and compute-done events.

- [ ] **Step 4: Verify GREEN across prefill and mixed stress**

Run: `/opt/miniconda3/bin/python -m pytest tests/unit/test_wave_stress.py tests/gpu/test_wave_numerics.py tests/gpu/test_wave_double_buffer.py -q`

Expected: exactness and event tests pass for unions through 128.

### Task 8: Real-Layer, Greedy E2E, Artifact, and Progress Gates

**Files:**
- Create: `scripts/run_correctness.py`
- Create: `scripts/run_graph_checks.py`
- Create: `src/vllm_latchmoe_cuda/artifacts.py`
- Test: `tests/unit/test_artifacts.py`
- Test: `tests/real/test_qwen_layers.py`
- Test: `tests/real/test_qwen_greedy.py`
- Modify: `docs/gpu_port/PROGRESS.md`

- [ ] **Step 1: Write failing artifact-label and real-test collection checks**

```python
def test_smoke_artifact_cannot_be_promoted(tmp_path):
    run = RunManifest(kind="smoke", repetitions=1)
    with pytest.raises(InvalidResultPromotionError):
        run.mark_final()

def test_real_layer_cases_cover_manifest(real_cases, frozen_manifest):
    assert {case.layer_id for case in real_cases} == set(frozen_manifest.layer_ids)
```

- [ ] **Step 2: Verify RED**

Run: `/opt/miniconda3/bin/python -m pytest tests/real --collect-only -q tests/unit/test_artifacts.py`

Expected: artifact helper and real runners are missing.

- [ ] **Step 3: Implement labeled runners and raw evidence inventory**

The runner writes `run_manifest.json`, command/environment, stdout/stderr,
correctness JSON, CUDA memory snapshots, profiler JSONL, and `SHA256SUMS`. It
supports `--kind smoke|warmup|measurement` and refuses a final label unless the
measurement protocol explicitly satisfies repetition requirements.

- [ ] **Step 4: Run complete verification and record actual status**

Run:

```bash
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration -q
/opt/miniconda3/bin/python -m pytest tests/gpu -q
/opt/miniconda3/bin/python -m pytest tests/real/test_qwen_layers.py -q
/opt/miniconda3/bin/python scripts/run_correctness.py --mode latchmoe-eager --kind smoke --artifact-dir artifacts/qwen-eager-smoke
/opt/miniconda3/bin/python scripts/run_correctness.py --mode latchmoe-piecewise --kind smoke --artifact-dir artifacts/qwen-piecewise-smoke
/opt/miniconda3/bin/python scripts/run_correctness.py --mode latchmoe-waves --kind smoke --artifact-dir artifacts/qwen-waves-smoke
/opt/miniconda3/bin/python scripts/run_correctness.py --mode uva --kind smoke --artifact-dir artifacts/qwen-uva-smoke
```

Expected: report each command's real exit status. GPU-unavailable, OOM, or API
failures remain failures/blockers with their logs; they are not converted to
passes. Append commands, artifacts, failures, and next steps to
`docs/gpu_port/PROGRESS.md`.

### Task 9: Final Review and Regression Gate

**Files:**
- Modify only files implicated by review findings.

- [ ] **Step 1: Run formatting and static checks**

Run:

```bash
/opt/miniconda3/bin/python -m ruff check .
/opt/miniconda3/bin/python -m ruff format --check .
/opt/miniconda3/bin/python -m pytest tests/unit tests/integration -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Review requirements against evidence**

Check every design invariant against a named test or artifact. Verify Stage 1,
Stage 2, and Stage 3 statuses separately and do not infer unrun GPU behavior from
CPU tests.

- [ ] **Step 3: Commit verified implementation**

```bash
git add pyproject.toml src tests benchmark scripts docs/gpu_port/PROGRESS.md
git commit -m "feat: implement CUDA LatchMoE stages"
```

Expected: local commit succeeds; no push is performed.
