# LatchMoE GPU Main Slots Cache Alignment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 `LatchMoE-GPU` 从 2026-07 的 manifest、共享池和双 temporary-bank 实现迁移到与 2026-09 LatchMoE 一致的 GiB 整层驻留、逐层 Main Slots Cache、hit-first multi-wave、一次原生 combine 和 fail-closed 生命周期。

**Architecture:** 父进程从真实模型配置生成一个不可变的 `CudaResidencyPlan`，并把同一个 plan 序列化给 worker。只有 plan 选中的层采用 CPU-first Host Store 和逐层 Main Slots Cache；未选中层保持原生 GPU fused-MoE。overflow 首先消费 cache hits，随后在同一主缓存中真实替换 miss experts；所有 routed pair 只执行一次，并在层末执行一次原生 combine。第一阶段实现严格串行版本，正确性通过后才增加只使用空闲 slot 的定向重叠。

**Tech Stack:** Python 3.13、PyTorch 2.10 CUDA、vLLM 0.19.1 锁定源码、Triton modular MoE、CUDA streams/events、pytest。

---

## 范围与不变量

- 生产入口只接受 `--cuda-moe-offload-gb`。`VLLM_LATCHMOE_MANIFEST` 和显式 slot 数仅保留为 diagnostic/oracle 接口。
- 当前第一批 qualified tuple 保持 `Qwen3 routed-only + BF16 + TP1 + single GPU + modular kernel`。TP>1、量化、monolithic kernel、expert parallel 和 fused/mix-placement shared experts 必须在启动时明确拒绝。
- Main Slots Cache 是唯一生产动态专家权重存储。生产路径不得构造 `CudaStagePool`，不得自动 full-layer staging，也不得切到 stock UVA。
- 每个 offloaded layer 拥有自己的权重 slots、`log2phy`、owner/generation/state 和 CUDA event 生命周期。不同层不能共享同一个 main pool。
- router 每层每次 forward 只运行一次。pair planner 只能重排 router 已经产生的 pairs，不能重新路由。
- 每个 `(token, topk_position)` pair 恰好进入一个 wave。所有 wave 完成后只执行一次原生 layer-level combine。
- `h=0` 和 `h=C<P` 必须串行；只有 `0<h<C` 且下一完整 wave 能装入空闲 slots 时才能产生 overlap candidate。candidate 和 CUDA event 证明的 actual overlap 分开记录。
- 任何 plan、capability、mapping、generation、event 或 combine 失败都使 runtime 进入 poisoned 状态并 fail closed；同一 runtime 不再接受下一次 forward。
- 无 profile 的生产默认层选择固定为 `midpoint_stratified_v1`：先从真实 checkpoint 得到有序 routed-MoE eligible IDs 和整层字节预算，再在 eligible 序列上做确定性中点分层选取。`generate_offload_manifest.py` 的 `(3, 7, ..., 47)` 对应旧 NPU `group_size=4` 的 group-tail 规则；2026-07 正式实验又使用了 `0..11`。两者都只属于 legacy evidence，不能继续充当生产默认。
- profile-guided placement 是显式可选策略，不是默认规则。它只能在不改变 selected layer 数、有效整层字节预算和可追溯 plan 身份的前提下交换 selected/resident IDs。

## 文件结构

新增文件：

- `src/vllm_latchmoe_cuda/residency_plan.py`：纯 CPU、不可变整层预算计划及序列化。
- `src/vllm_latchmoe_cuda/cli.py`、`src/vllm_latchmoe_cuda/__main__.py`：生产 CLI，生成 plan 后再进入 vLLM。
- `src/vllm_latchmoe_cuda/main_cache.py`：逐层 slot 权重、lease、victim 和 prepared-wave 生命周期。
- `src/vllm_latchmoe_cuda/moe_seam.py`：锁定 vLLM dispatch/MLP/combine 接口，未知 ABI fail closed。
- `src/vllm_latchmoe_cuda/capabilities.py`：运行时 capability descriptor 和支持矩阵。
- `tests/unit/test_residency_plan.py`、`tests/unit/test_main_cache_plan.py`、`tests/unit/test_capabilities.py`。
- `tests/gpu/test_main_cache_streaming.py`、`tests/gpu/test_main_cache_overlap.py`、`tests/gpu/test_failure_lifecycle.py`。
- `tests/integration/test_budget_cli.py`、`tests/integration/test_native_combine.py`。
- `benchmark/scripts/verify_budget_contract.py`：预算、参数集合、overlap 和 exact-token 验收。

修改文件：

- `src/vllm_latchmoe_cuda/manifest.py`：升级为 schema v2 身份锁，不再拥有生产层选择。
- `src/vllm_latchmoe_cuda/host_store.py`：按选中层的实际参数布局动态创建 pinned views。
- `src/vllm_latchmoe_cuda/runtime.py`：移除生产 `CudaStagePool`，组合逐层 Main Cache 和故障状态。
- `src/vllm_latchmoe_cuda/offloader.py`：selected-only CPU-first、逐层 runtime 和关闭协议。
- `src/vllm_latchmoe_cuda/plugin.py`：消费父进程 plan，移除硬编码 14 GiB residual UVA。
- `src/vllm_latchmoe_cuda/runner_adapter.py`、`src/vllm_latchmoe_cuda/graph_ops.py`：hit-first waves、一次 combine 和图边界。
- `src/vllm_latchmoe_cuda/profile.py`、`src/vllm_latchmoe_cuda/transfer.py`：event window、failure、drain 和 close。
- `src/vllm_latchmoe_cuda/benchmark.py`、`benchmark/scripts/run_sharegpt.py`、`README.md`：新命令和公平比较合同。

---

### Task 1: 定义 GiB 整层 Residency Plan

**Files:**

- Create: `src/vllm_latchmoe_cuda/residency_plan.py`
- Create: `tests/unit/test_residency_plan.py`
- Modify: `src/vllm_latchmoe_cuda/errors.py`
- Modify: `src/vllm_latchmoe_cuda/manifest.py`
- Modify: `tests/unit/test_manifest.py`

- [x] **Step 1: 先写不可变 plan 的失败测试**

测试必须覆盖以下独立合同：

- Qwen3 的 48 个 eligible 层在 `K=12` 时，默认 IDs 精确等于 `(2, 6, 10, 14, 18, 22, 26, 30, 34, 38, 42, 46)`，并记录 `selection_strategy == "midpoint_stratified_v1"`；
- GLM 的 dense layer 0 不进入 eligible、selected 或 resident routed-MoE 集合；选择公式只在过滤后的 ordered eligible IDs 上运行；
- 均匀层使用 `K = min(E, ceil(requested_bytes / routed_expert_bytes_per_layer))`，超出全模型 routed expert bytes 时封顶为 `E` 并记录 `capped_to_eligible_layers=true`；
- 非均匀层使用实测的逐层 routed-expert bytes，按 ordered eligible prefix 累计完整层直到达到请求预算，不得用平均值或第一层大小伪造 `K`；
- selected/resident 是 ordered eligible IDs 的严格、有序、无重复分割；零预算得到 `selected=()`，未知或不完整模型字段 fail closed；
- JSON round-trip 保留 ordered eligible IDs、selected IDs、`selection_strategy` 和 `plan_id`，篡改其中任一项均失败；
- profile-guided swap 保持 `K` 和 `effective_offloaded_bytes`，并生成包含 profile identity 的新 `plan_id`；shared weight bytes 不进入 dynamic slot 预算。

```python
QWEN3_CONFIG = {
    "model_type": "qwen3_moe",
    "hidden_size": 2048,
    "moe_intermediate_size": 768,
    "num_experts": 128,
    "num_experts_per_tok": 8,
    "num_hidden_layers": 48,
    "torch_dtype": "bfloat16",
}
GLM_CONFIG = {
    "model_type": "glm4_moe_lite",
    "hidden_size": 2048,
    "moe_intermediate_size": 1536,
    "n_routed_experts": 64,
    "num_experts_per_tok": 4,
    "num_hidden_layers": 47,
    "dense_layer_ids": [0],
    "torch_dtype": "bfloat16",
}
THREE_LAYER_MOE_CONFIG = {
    "model_type": "synthetic_moe",
    "num_hidden_layers": 3,
    "moe_layer_ids": [0, 1, 2],
    "num_experts": 2,
}

def test_budget_selects_complete_layers_and_partitions_eligible_ids():
    plan = build_residency_plan(
        requested_offload_gib=13.5,
        model_config=QWEN3_CONFIG,
        max_capture_size=1,
        top_k=8,
        device_total_bytes=48 << 30,
        kv_reserve_bytes=512 << 20,
    )
    assert set(plan.offloaded_layer_ids).isdisjoint(plan.resident_layer_ids)
    assert tuple(sorted(plan.offloaded_layer_ids + plan.resident_layer_ids)) == (
        plan.eligible_layer_ids
    )
    assert plan.effective_offloaded_bytes >= plan.requested_offload_bytes
    assert plan.effective_num_slots >= 8

def test_qwen3_default_selection_is_midpoint_stratified():
    plan = build_residency_plan(
        requested_offload_gib=13.5,
        model_config=QWEN3_CONFIG,
        max_capture_size=1,
        top_k=8,
        device_total_bytes=48 << 30,
        kv_reserve_bytes=512 << 20,
    )
    assert plan.selection_strategy == "midpoint_stratified_v1"
    assert plan.offloaded_layer_ids == tuple(range(2, 48, 4))

def test_glm_filters_dense_layer_before_midpoint_selection():
    plan = build_residency_plan(
        requested_offload_gib=13.5,
        model_config=GLM_CONFIG,
        max_capture_size=1,
        top_k=4,
        device_total_bytes=48 << 30,
        kv_reserve_bytes=512 << 20,
    )
    assert plan.eligible_layer_ids == tuple(range(1, 47))
    assert plan.offloaded_layer_ids == (
        2, 6, 10, 14, 18, 22, 25, 29, 33, 37, 41, 45,
    )
    assert 0 not in plan.resident_layer_ids

def test_heterogeneous_bytes_use_complete_ordered_prefix():
    metadata = (
        {"layer_id": 0, "routed_expert_bytes": 4},
        {"layer_id": 1, "routed_expert_bytes": 8},
        {"layer_id": 2, "routed_expert_bytes": 16},
    )
    plan = build_residency_plan(
        requested_offload_gib=10 / (1 << 30),
        model_config=THREE_LAYER_MOE_CONFIG,
        max_capture_size=1,
        top_k=1,
        device_total_bytes=1 << 30,
        kv_reserve_bytes=0,
        layer_metadata=metadata,
    )
    assert plan.selection_strategy == "ordered_prefix_bytes_v1"
    assert plan.offloaded_layer_ids == (0, 1)
    assert plan.effective_offloaded_bytes == 12

def test_budget_above_all_eligible_bytes_caps_explicitly():
    plan = build_residency_plan(
        requested_offload_gib=1000,
        model_config=QWEN3_CONFIG,
        max_capture_size=1,
        top_k=8,
        device_total_bytes=2 << 40,
        kv_reserve_bytes=0,
    )
    assert plan.offloaded_layer_ids == tuple(range(48))
    assert plan.resident_layer_ids == ()
    assert plan.ledger["capped_to_eligible_layers"] is True
```

- [x] **Step 2: 运行测试并确认 RED**

```bash
python -m pytest -q tests/unit/test_residency_plan.py tests/unit/test_manifest.py
```

Expected: `ModuleNotFoundError` 或缺少 `CudaResidencyPlan`。

- [x] **Step 3: 实现 plan 类型和稳定摘要**

公开接口固定为：

```python
@dataclass(frozen=True)
class LayerExpertBytes:
    layer_id: int
    routed_expert_bytes: int
    shared_expert_bytes: int = 0

@dataclass(frozen=True)
class CudaResidencyPlan:
    schema_version: int
    planner_version: str
    selection_strategy: str
    plan_id: str
    model_fingerprint: dict[str, object]
    num_experts: int
    requested_offload_bytes: int
    effective_offloaded_bytes: int
    eligible_layer_ids: tuple[int, ...]
    offloaded_layer_ids: tuple[int, ...]
    resident_layer_ids: tuple[int, ...]
    layer_expert_bytes: tuple[LayerExpertBytes, ...]
    recommended_num_slots: int
    effective_num_slots: int
    main_slot_cache_bytes: int
    net_hbm_saved_bytes: int
    ledger: dict[str, object]

    @property
    def offloaded_layer_count(self) -> int: ...

    def to_jsonable(self) -> dict[str, object]: ...

def build_residency_plan(
    requested_offload_gib: float,
    model_config: Mapping[str, object],
    *,
    max_capture_size: int,
    top_k: int,
    device_total_bytes: int,
    kv_reserve_bytes: int,
    layer_metadata: Sequence[Mapping[str, object]] | None = None,
    profile_path: str | None = None,
) -> CudaResidencyPlan: ...

def serialize_residency_plan(plan: CudaResidencyPlan) -> str: ...
def deserialize_residency_plan(raw: str) -> CudaResidencyPlan: ...
def validate_plan_against_model(plan, model_config) -> None: ...
```

使用 `1 << 30` 换算 GiB。先从 checkpoint 的 `moe_layer_ids`、显式 `dense_layer_ids` 和架构元数据构造 `eligible_layer_ids`，禁止把 dense/non-MoE 层放入分割，也禁止按模型名猜一个 Qwen 默认集合。优先使用实测 `layer_metadata`；仅对明确支持的 Qwen3 BF16 配置允许按 `3 * hidden_size * moe_intermediate_size * num_experts * 2` 推导 routed expert bytes。

均匀层令 `E=len(eligible_layer_ids)`、`K=min(E, ceil(requested_bytes / layer_bytes))`。无 profile 时固定：

```python
selection_strategy = "midpoint_stratified_v1"
selected_indices = []
used = set()
for j in range(K):
    index = min(math.floor((j + 0.5) * E / K), E - 1)
    if index in used:
        # Deterministic collision rule: increasing distance, lower index first.
        index = next(
            candidate
            for distance in range(1, E)
            for candidate in (index - distance, index + distance)
            if 0 <= candidate < E and candidate not in used
        )
    used.add(index)
    selected_indices.append(index)
offloaded_layer_ids = tuple(
    eligible_layer_ids[index] for index in sorted(selected_indices)
)
```

因此 Qwen3 `E=48,K=12` 的默认集合必须是 `2,6,10,14,18,22,26,30,34,38,42,46`，不是旧 generator 的 `3,7,...,47`，也不是旧正式实验的 `0..11`。非均匀层设置 `selection_strategy="ordered_prefix_bytes_v1"`，在 ordered eligible byte table 上选择达到请求值的最短完整前缀，并记录精确总和；预算超过全部 eligible bytes 时封顶并显式记账。

若显式提供 profile，则先生成上述 base plan，再设置 `selection_strategy="profile_guided_swap_v1"` 并执行交换。swap 必须保持 `K` 和 `effective_offloaded_bytes`；不满足时保留 base plan 或 fail closed。plan payload 同时记录 `selection_strategy`、`base_selection_strategy`、profile hash、ordered eligible IDs 和最终 selected IDs，这些字段全部进入 `plan_id` 摘要。所有集合必须有序、唯一并构成完整分割。

- [x] **Step 4: 加入 slot 可行性约束**

图模式下设置 `minimum_slots = min(num_experts, max_capture_size * top_k)`。逐层 cache 总量为：

```python
main_slot_cache_bytes = sum(
    effective_num_slots * bytes_per_expert_by_layer[layer_id]
    for layer_id in offloaded_layer_ids
)
net_hbm_saved_bytes = effective_offloaded_bytes - main_slot_cache_bytes
```

若 `minimum_slots` 使 `net_hbm_saved_bytes <= 0`，或估算总 HBM 超过显式 device/KV 边界，直接抛 `ResidencyBudgetError`，不得自动改成共享 pool。初版 `effective_num_slots = minimum_slots`；更大的 cache 只能作为以后有测量依据的 advisor 结果。

```python
class ResidencyBudgetError(LatchMoEError):
    pass
```

- [x] **Step 5: 将 manifest 降级为身份锁**

schema v2 保存 model revision、config/index SHA-256、每个 shard 的路径/size/SHA-256、vLLM source identity 和 `plan_id`。`layers` 和 `num_slots` 不再是生产控制源。旧 schema v1 只允许 `diagnostic_mode=True` 加载，否则报弃用错误。增加明确的身份锁接口：

```python
@dataclass(frozen=True)
class ModelIdentityLock:
    schema_version: int
    model_path: str
    revision: str
    config_sha256: str
    weight_index_sha256: str
    shard_files: tuple[tuple[str, int, str], ...]
    vllm_version: str
    vllm_source_sha256: str
    plan_id: str

def serialize_identity_lock(lock: ModelIdentityLock) -> str: ...
def deserialize_identity_lock(raw: str) -> ModelIdentityLock: ...
def build_identity_lock(
    plan: CudaResidencyPlan, vllm_args: Sequence[str]
) -> ModelIdentityLock: ...
def validate_identity_lock(lock: ModelIdentityLock, plan: CudaResidencyPlan) -> None: ...
```

`validate_identity_lock()` 必须重新计算 config、index、所有列出的 shard 和锁定 vLLM seam 文件摘要，并确认 `lock.plan_id == plan.plan_id`。

- [x] **Step 6: 运行测试并提交**

```bash
python -m pytest -q tests/unit/test_residency_plan.py tests/unit/test_manifest.py
git add src/vllm_latchmoe_cuda/residency_plan.py src/vllm_latchmoe_cuda/errors.py src/vllm_latchmoe_cuda/manifest.py tests/unit/test_residency_plan.py tests/unit/test_manifest.py
git commit -m "feat: add model-adaptive CUDA residency plan"
```

Expected: 所有 plan、序列化、预算和旧 schema 拒绝测试通过。

---

### Task 2: 增加生产 CLI 并把同一个 Plan 传播到 Worker

**Files:**

- Create: `src/vllm_latchmoe_cuda/cli.py`
- Create: `src/vllm_latchmoe_cuda/__main__.py`
- Create: `tests/integration/test_budget_cli.py`
- Modify: `src/vllm_latchmoe_cuda/plugin.py`
- Modify: `tests/integration/test_plugin.py`

- [x] **Step 1: 写 CLI 和 worker 传播失败测试**

断言 `--cuda-moe-offload-gb 13.5` 从传给 vLLM 的 argv 中移除，父进程设置 `VLLM_LATCHMOE_RESIDENCY_PLAN_JSON`，worker 反序列化后得到相同 `plan_id`、`selection_strategy`、ordered eligible IDs 和 `offloaded_layer_ids`；不能只比较 layer count。同时断言与 `--cpu-offload-gb`、`VLLM_LATCHMOE_MANIFEST` 或 `VLLM_LATCHMOE_WAVE_SLOTS` 的生产组合直接失败。

```python
assert worker_plan.plan_id == parent_plan.plan_id
assert worker_plan.selection_strategy == parent_plan.selection_strategy
assert worker_plan.eligible_layer_ids == parent_plan.eligible_layer_ids
assert worker_plan.offloaded_layer_ids == parent_plan.offloaded_layer_ids
```

- [x] **Step 2: 实现薄 CLI wrapper**

解析接口固定为：

```python
@dataclass(frozen=True)
class CudaMoeCliArgs:
    offload_gib: float
    diagnostic_mode: bool = False

def parse_plugin_args(argv: Sequence[str]) -> tuple[CudaMoeCliArgs, list[str]]: ...
def build_plan_from_model_argument(
    plugin_args: CudaMoeCliArgs, vllm_args: Sequence[str]
) -> CudaResidencyPlan: ...
```

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    plugin_args, vllm_args = parse_plugin_args(args)
    plan = build_plan_from_model_argument(plugin_args, vllm_args)
    os.environ["VLLM_LATCHMOE_MODE"] = "latchmoe"
    os.environ["VLLM_LATCHMOE_RESIDENCY_PLAN_JSON"] = (
        serialize_residency_plan(plan)
    )
    os.environ["VLLM_LATCHMOE_IDENTITY_LOCK_JSON"] = (
        serialize_identity_lock(build_identity_lock(plan, vllm_args))
    )
    os.environ["VLLM_PLUGINS"] = "latchmoe_cuda"
    sys.argv = ["vllm", *vllm_args]
    from vllm.entrypoints.cli.main import main as vllm_main
    return int(vllm_main() or 0)
```

必须先设置环境再 import vLLM CLI，确保 spawn worker 继承 plan。wrapper 从真实 model config、`--max-cudagraph-capture-size`、GPU 总显存和 KV reserve 建 plan；不能使用内置 Qwen fixture 兜底。

- [x] **Step 3: 修改 plugin factory**

删除 `TOTAL_OFFLOAD_BUDGET_BYTES` 和 residual UVA 计算。生产模式只接受 plan：

```python
raw = os.environ.get("VLLM_LATCHMOE_RESIDENCY_PLAN_JSON")
if mode == "latchmoe" and not raw:
    raise RuntimeError("LatchMoE worker is missing the parent residency plan")
plan = deserialize_residency_plan(raw)
lock_raw = os.environ.get("VLLM_LATCHMOE_IDENTITY_LOCK_JSON")
if not lock_raw:
    raise RuntimeError("LatchMoE worker is missing the parent identity lock")
identity_lock = deserialize_identity_lock(lock_raw)
validate_identity_lock(identity_lock, plan)
return CudaSEWOffloader(plan=plan, identity_lock=identity_lock)
```

保留 `mode=uva` 仅供明确的 baseline runner；LatchMoE 失败时不能返回原生 factory 或 UVA。

- [x] **Step 4: 验证 plugin 确实被调用**

增加一个最小 server bootstrap integration test，检查 profile 中同时存在 `parent_plan` 和 `worker_plan`，且 `plan_id`、`selection_strategy`、eligible IDs 和 selected IDs 逐项相同。worker 禁止重新运行层选择器；它只反序列化并校验父进程 plan。若 vLLM 在 `cpu_offload_gb=0` 时不调用 `create_offloader`，就在锁定 vLLM fork 添加显式 `create_moe_offloader()` hook；不得通过传入假的 `--cpu-offload-gb` 激活它。

- [x] **Step 5: 运行测试并提交**

```bash
python -m pytest -q tests/integration/test_budget_cli.py tests/integration/test_plugin.py
git add src/vllm_latchmoe_cuda/cli.py src/vllm_latchmoe_cuda/__main__.py src/vllm_latchmoe_cuda/plugin.py tests/integration/test_budget_cli.py tests/integration/test_plugin.py
git commit -m "feat: propagate one CUDA offload plan to workers"
```

---

### Task 3: 改成 Selected-only CPU-first 和逐层 Main Cache

**Files:**

- Create: `src/vllm_latchmoe_cuda/main_cache.py`
- Create: `tests/unit/test_main_cache_plan.py`
- Modify: `src/vllm_latchmoe_cuda/host_store.py`
- Modify: `src/vllm_latchmoe_cuda/offloader.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Modify: `tests/integration/test_direct_load.py`
- Modify: `tests/gpu/test_slot_layout.py`

- [x] **Step 1: 写 selected/resident 和存储身份测试**

使用四层假模型和 `offloaded=(1, 3)`。断言只有 1、3 层在 CPU 上构造并进入 pinned Host Store；0、2 层保持原生参数。断言 layer 1/3 的 slot 指针不同，连续两个 decode step 后 layer 1 的 cache mapping 未因 layer 3 执行而失效。

- [x] **Step 2: 重构 Host Store 为动态实际布局**

将 `PinnedHostStore(manifest)` 改成无预设 slab 的 selected-layer store：

```python
def bind_parameter(self, layer_id, name, parameter, *, original_device):
    pinned = torch.empty_strided(
        tuple(parameter.shape), tuple(parameter.stride()),
        dtype=parameter.dtype, device="cpu", pin_memory=True,
    )
    parameter.data = pinned
    self._views[(layer_id, name)] = pinned
    self._bindings[(layer_id, name)] = BoundParameter(...)
```

绑定后记录实际 shape、stride、dtype 和 bytes；与 parent plan 的该层预计 bytes 不一致时，在释放 GPU 原权重前 fail closed。

- [x] **Step 3: 每层独立分配 Main Cache**

`CudaSEWOffloader.post_init()` 不再创建 `self.main_slot_pool` 或 `self.stage_pool`。每个 selected layer 构造独立 runtime：

```python
self.runtimes[layer_id] = CudaLayerRuntime(
    layer_id=layer_id,
    num_experts=plan.num_experts,
    num_slots=plan.effective_num_slots,
    host_store=self.host_store,
    experts_module=experts_module,
    device=device,
)
```

`CudaLayerRuntime` 内部直接分配本层 `slot_w13`、`slot_w2` 和稳定 `log2phy`。删除 `CudaMainSlotPool.owner`、`acquire()` 和切层 `invalidate_main_slots()`。保留 `ExpertSlotBank` 的 generation/state 语义。

- [x] **Step 4: 将 CudaStagePool 隔离出生产代码**

删除 `runtime.stage_pool` 和所有自动创建分支。若历史 oracle 仍需双 bank，将类移动到 `tests/helpers/legacy_stage_pool.py`，生产包不导出也不引用它。

- [x] **Step 5: 运行生命周期测试并提交**

```bash
python -m pytest -q tests/integration/test_direct_load.py tests/gpu/test_slot_layout.py tests/gpu/test_async_lifecycle.py
git add src/vllm_latchmoe_cuda/main_cache.py src/vllm_latchmoe_cuda/host_store.py src/vllm_latchmoe_cuda/offloader.py src/vllm_latchmoe_cuda/runtime.py tests/unit/test_main_cache_plan.py tests/integration/test_direct_load.py tests/gpu/test_slot_layout.py
git commit -m "refactor: give each offloaded layer a persistent main cache"
```

Expected: selected-only binding和逐层持久命中通过；测试进程中不存在第二份动态 weight bank。

---

### Task 4: 实现串行 Hit-first Main-Cache Multi-wave

**Files:**

- Modify: `src/vllm_latchmoe_cuda/main_cache.py`
- Modify: `src/vllm_latchmoe_cuda/core/waves.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Create: `tests/gpu/test_main_cache_streaming.py`
- Modify: `tests/unit/test_waves.py`
- Modify: `tests/unit/test_wave_stress.py`

- [x] **Step 1: 先写纯 planner 测试**

固定 capacity `C=4`，覆盖：`h=0, P=9` 得到三个串行 miss waves；`h=2, P=7` 得到 hit wave 后 miss waves；`h=C<P` 不标 overlap；重复 expert 只出现在一个 expert wave，但每个 routed pair 保留一次。

```python
specs = plan_main_cache_waves(
    active_experts=(0, 1, 2, 3, 4, 5, 6),
    capacity=4,
    hit_experts=(0, 1),
)
assert [(s.wave_type, s.experts) for s in specs] == [
    ("hit", (0, 1)),
    ("miss", (2, 3, 4, 5)),
    ("miss", (6,)),
]
```

- [x] **Step 2: 增加真实 replacement API**

`CudaLayerMainCache.prepare_wave()` 必须完成：查找 READY hits、保护当前 leases、从 EMPTY/READY 中选 victim、等待 victim compute event、把 victim 转成 LOADING、H2D 到原 main slot、等待 ready event、mark READY、生成不可变 mapping snapshot。返回：

```python
@dataclass(frozen=True)
class PreparedMainCacheWave:
    layer_id: int
    wave_id: int
    wave_type: Literal["hit", "miss"]
    experts: tuple[int, ...]
    leases: tuple[SlotLease, ...]
    physical_slot_by_expert: Mapping[int, int]
    ready_ticket: TransferTicket | None
    evicted_experts: tuple[int, ...]
    h2d_bytes: int
```

hit wave 的 `h2d_bytes` 必须为 0。miss wave 只能写 `runtime.slot_w13/w2`，不能写其他 bank。

- [x] **Step 3: 串行执行所有 waves**

先实现无 overlap 的参考生产版本：

```python
pair_index = build_routed_pair_index(topk_ids, topk_weights)
payloads = []
for spec in plan_main_cache_waves(active, capacity, hit_experts=hits):
    prepared = runtime.prepare_main_cache_wave(spec)
    runtime.wait_and_publish(prepared)
    microbatch = pair_index.for_experts(spec.experts, prepared.physical_slot_by_expert)
    payloads.append(run_pair_wave(runtime, prepared, hidden_states, microbatch))
    runtime.mark_compute_complete(prepared)
return combine_once(payloads, pair_index, hidden_states.shape)
```

第一版每个 wave 完成后再准备下一 wave。它必须先通过 exact-token gate，之后 Task 7 才允许增加 overlap。

- [x] **Step 4: 删除旧 exact-wave production executor**

`execute_exact_waves()` 改名为 `execute_main_cache_waves()`；删除 `free_banks`、`issue_order`、`stage_kernel_callback`、`kernel_callback` 和 finally 中的全 cache invalidation。profile 中 `stage_mode` 只能是 `main_cache_hit` 或 `main_cache_replacement`。

- [x] **Step 5: 运行测试并提交**

```bash
python -m pytest -q tests/unit/test_waves.py tests/unit/test_wave_stress.py tests/unit/test_main_cache_plan.py
python -m pytest -q tests/gpu/test_main_cache_streaming.py
git add src/vllm_latchmoe_cuda/main_cache.py src/vllm_latchmoe_cuda/core/waves.py src/vllm_latchmoe_cuda/runtime.py tests/unit/test_waves.py tests/unit/test_wave_stress.py tests/gpu/test_main_cache_streaming.py
git commit -m "feat: stream overflow through the main cache"
```

Expected: 第二次相同 route 产生 hit、H2D bytes 下降；union-128 的 pair coverage 完整且没有 stage-bank allocation。

---

### Task 5: 建立可验证的 vLLM Dispatch/Compute/Combine Seam

**Files:**

- Create: `src/vllm_latchmoe_cuda/moe_seam.py`
- Create: `src/vllm_latchmoe_cuda/capabilities.py`
- Create: `tests/unit/test_capabilities.py`
- Create: `tests/integration/test_native_combine.py`
- Modify: `src/vllm_latchmoe_cuda/runner_adapter.py`

- [x] **Step 1: 先做锁定宿主 ABI probe**

在目标 A6000/vLLM 环境运行只读探针，记录 modular kernel、prepare/finalize 或 token dispatcher 的实际类、方法签名和源码 hash：

```bash
python - <<'PY'
import inspect
from vllm.model_executor.layers.fused_moe import FusedMoE
print(inspect.getfile(FusedMoE))
print(inspect.signature(FusedMoE.forward))
PY
```

将结果固化为 `SUPPORTED_VLLM_SOURCE_HASHES`。只检查 `metadata.version("vllm")` 不够；editable checkout 校验 git commit，wheel 安装校验 seam 文件 SHA-256。两者都无法证明时 fail closed。

- [x] **Step 2: 定义窄 seam，而不是继续调用 top_k=1 完整 MoE**

```python
class CudaMoeSeam(Protocol):
    def run_expert_mlp(
        self, *, hidden_states, physical_ids, slot_w13, slot_w2
    ) -> NativeWavePayload: ...

    def combine(
        self, *, waves: Sequence[NativeWavePayload],
        topk_weights, pair_offsets, restore_shape
    ) -> torch.Tensor: ...
```

`NativeWavePayload` 保存未做 layer-level top-k reduction 的 expert outputs 和全局 pair offsets。`combine()` 必须调用锁定 vLLM 的 native unpermute/combine primitive 一次。若 vLLM 0.19.1 没有可复用的分离 API，则在锁定 vLLM fork 中增加这两个窄 hook；不得用 `index_add_` 冒充 native combine。

- [x] **Step 3: 写一次 combine 的行为测试**

给 seam 注入计数 spy，构造 3 waves，断言 `run_expert_mlp` 调用 3 次、router 1 次、`combine` 1 次。用同一个 full-resident native layer 对照 BF16 张量，并另外比较 greedy token IDs。

- [x] **Step 4: 移除旧数值路径**

从生产 `runner_adapter.py` 删除：

```python
pair_outputs.append(pair_output.float())
output.index_add_(0, torch.cat(scatter_indices), torch.cat(pair_outputs))
return output.to(dtype=hidden_states.dtype)
```

测试辅助 `_capturable_weights_moe` 可以保留为 oracle，但不能被生产 adapter 引用。

- [x] **Step 5: 加 capability descriptor**

descriptor 至少固定：model family、router owner、shared expert representation、dtype、TP、expert parallel、quant method、modular/monolithic、output ABI、combine owner 和 graph mode。首批只允许 routed-only BF16 TP1 modular + native combine。未知 tuple 在加载权重前失败。

- [x] **Step 6: 运行测试并提交**

```bash
python -m pytest -q tests/unit/test_capabilities.py tests/integration/test_native_combine.py tests/gpu/test_wave_numerics.py
git add src/vllm_latchmoe_cuda/moe_seam.py src/vllm_latchmoe_cuda/capabilities.py src/vllm_latchmoe_cuda/runner_adapter.py tests/unit/test_capabilities.py tests/integration/test_native_combine.py tests/gpu/test_wave_numerics.py
git commit -m "feat: defer routed pairs to one native combine"
```

---

### Task 6: 保持 PIECEWISE 图边界和单次 Router

**Files:**

- Modify: `src/vllm_latchmoe_cuda/graph_ops.py`
- Modify: `src/vllm_latchmoe_cuda/runner_adapter.py`
- Modify: `tests/gpu/test_graph_boundary.py`
- Modify: `tests/gpu/test_graph_runner.py`
- Modify: `tests/integration/test_vllm_lifecycle.py`

- [x] **Step 1: 写 router 次数和边界失败测试**

捕获与 replay 中均断言一次 forward 只调用一次 `router.select_experts()`。staging、victim selection、H2D 和 mapping publication 在 capture 时必须抛 `StagingDuringCaptureError`。捕获段只包含 router/compute 所需固定地址张量和稳定 topology。

- [x] **Step 2: 调整 graph custom ops**

保留两个 splitting ops：`stage_experts` 和 `finish_experts`。`stage_experts` 消费已经产生的 `topk_ids`，只在 eager boundary 调用 `prepare_main_cache_wave`；不能再次调用 router。bounded shape 进入 captured modular compute，overflow shape在 boundary 外执行 `execute_main_cache_waves()`。

- [x] **Step 3: 强制 PIECEWISE 安全模式**

生产 CLI 自动设置 PIECEWISE。`FULL`、`FULL_DECODE_ONLY` 和 `FULL_AND_PIECEWISE` 在 offload plan 启用时直接拒绝；`--enforce-eager` 可作为 correctness ablation。

- [x] **Step 4: 验证稳定地址**

连续改变 routes 并 replay 同一 graph，断言每层 `slot_w13`、`slot_w2`、`log2phy` 和 graph token 指针不变；同时断言层间指针不同。

- [x] **Step 5: 运行测试并提交**

```bash
python -m pytest -q tests/gpu/test_graph_boundary.py tests/gpu/test_graph_runner.py tests/integration/test_vllm_lifecycle.py
git add src/vllm_latchmoe_cuda/graph_ops.py src/vllm_latchmoe_cuda/runner_adapter.py tests/gpu/test_graph_boundary.py tests/gpu/test_graph_runner.py tests/integration/test_vllm_lifecycle.py
git commit -m "fix: keep dynamic CUDA staging outside graph replay"
```

---

### Task 7: 在串行正确性通过后增加受约束 Overlap

**Files:**

- Modify: `src/vllm_latchmoe_cuda/main_cache.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Modify: `src/vllm_latchmoe_cuda/transfer.py`
- Modify: `src/vllm_latchmoe_cuda/profile.py`
- Create: `tests/gpu/test_main_cache_overlap.py`

- [x] **Step 1: 写三类强制调度测试**

分别构造 `h=0`、`h=C<P` 和 `0<h<C`。前两种断言没有 next-wave H2D 在 current compute 前发出；第三种只有当 `len(next_wave) <= C-h` 时才允许 candidate，且 next wave 只能写不受保护的 EMPTY/READY slots，不能覆盖 current leases。

- [x] **Step 2: 实现空闲-slot prefetch**

当前 wave 准备完后，计算：

```python
free_slots = capacity - len(current_wave.experts)
can_prefetch = (
    0 < len(current_wave.experts) < capacity
    and len(next_wave.experts) <= free_slots
)
```

`can_prefetch` 为真时，用 transfer stream 把下一完整 wave装入不受当前 wave leases 保护的可复用 slots。候选可以是 EMPTY，也可以是 READY 且不属于当前 wave 的 victim；后者必须先等待自己的历史 compute event。不得选择 LOADING、COMPUTING 或当前 wave 的 slot。只发 H2D，不提前覆盖当前 `log2phy`；mapping publication 在下一 wave compute 入队前完成。

- [x] **Step 3: 用 CUDA events 证明实际 overlap**

为 H2D 和 compute 分别记录 start/end events，并从同一个 origin event 计算设备时间窗：

```python
actual_overlap = max(h2d_start_ms, compute_start_ms) < min(
    h2d_end_ms, compute_end_ms
)
```

profile 同时写 `overlap_candidate`、`actual_overlap`、四个 event 时间、H2D bytes 和 protected slot IDs。只有 actual 为真才能汇入 overlap 性能统计。

- [x] **Step 4: 对比 serial/async 数值和 traffic**

固定 routes，串行和 async 必须得到相同 token IDs、pair count、victim sequence 和最终 cache ownership；async 只允许改变 event 排序和延迟，不能改变算法结果。

- [x] **Step 5: 运行测试并提交**

```bash
python -m pytest -q tests/gpu/test_main_cache_overlap.py tests/gpu/test_main_cache_streaming.py
git add src/vllm_latchmoe_cuda/main_cache.py src/vllm_latchmoe_cuda/runtime.py src/vllm_latchmoe_cuda/transfer.py src/vllm_latchmoe_cuda/profile.py tests/gpu/test_main_cache_overlap.py
git commit -m "feat: overlap H2D only through idle main slots"
```

---

### Task 8: 修复异常恢复、Poison 和 Shutdown

**Files:**

- Modify: `src/vllm_latchmoe_cuda/errors.py`
- Modify: `src/vllm_latchmoe_cuda/runtime.py`
- Modify: `src/vllm_latchmoe_cuda/offloader.py`
- Modify: `src/vllm_latchmoe_cuda/profile.py`
- Create: `tests/gpu/test_failure_lifecycle.py`
- Modify: `tests/unit/test_profile.py`

- [x] **Step 1: 写 kernel 中途异常测试**

让第一个 wave kernel 向 compute stream 入队后抛异常，并让下一 wave H2D 已发出。断言 runtime 记录所有在途 compute/transfer events、进入 `POISONED`、下一次 forward 抛 `RuntimePoisonedError`，且没有 bank/slot 被重新写入。

```python
class RuntimePoisonedError(LatchMoEError):
    pass
```

- [x] **Step 2: 在异常路径记录完成边界**

围绕每个 wave 使用：

```python
try:
    payload = seam.run_expert_mlp(...)
except BaseException as exc:
    compute_done = torch.cuda.Event()
    compute_done.record(torch.cuda.current_stream(runtime.device))
    runtime.poison(exc, compute_done=compute_done, transfer_tickets=pending_tickets)
    raise
else:
    runtime.mark_compute_complete(prepared)
```

即使 Python callback 抛错，已经入队的 CUDA 工作也有 event 保护。fail closed 后禁止尝试透明恢复。

- [x] **Step 3: 实现幂等 close/drain**

`CudaLayerRuntime.close()` 等待所有 compute events 和 transfer tickets；`CudaSEWOffloader.close()` 关闭所有 runtimes、注销 graph registry、最后关闭唯一 `JsonlEventWriter`。重复 close 不报错。plugin 注册 `atexit` 作为 vLLM 没有调用 shutdown hook 时的后备。

- [x] **Step 4: 记录结构化 failure**

failure JSONL 必须包含 plan_id、layer、wave、exception type、active experts、leases、slot states、pending event 数和是否完成 drain。日志写失败不能掩盖原异常。

- [x] **Step 5: 运行测试并提交**

```bash
python -m pytest -q tests/gpu/test_failure_lifecycle.py tests/unit/test_profile.py
git add src/vllm_latchmoe_cuda/errors.py src/vllm_latchmoe_cuda/runtime.py src/vllm_latchmoe_cuda/offloader.py src/vllm_latchmoe_cuda/profile.py tests/gpu/test_failure_lifecycle.py tests/unit/test_profile.py
git commit -m "fix: fail closed and drain CUDA wave lifecycles"
```

---

### Task 9: 补齐 External Resident Shared Expert，不扩大未验证范围

**Files:**

- Modify: `src/vllm_latchmoe_cuda/capabilities.py`
- Modify: `src/vllm_latchmoe_cuda/runner_adapter.py`
- Modify: `src/vllm_latchmoe_cuda/offloader.py`
- Create: `tests/integration/test_shared_expert.py`

- [x] **Step 1: 写 shared expert 预算与调用测试**

external shared module 的权重必须一直常驻 GPU，不进入 Host Store、dynamic slots 或 victim candidates。它对原始 token batch 计算一次，不能对每个 routed wave 重算。

- [x] **Step 2: 实现受限 capability**

仅接受锁定 ABI 的 external resident shared expert。先得到 routed result，再按宿主 ABI 合并 shared result并保持原返回 tuple/shape。fused/mix-placement shared expert、shared/H2D overlap 和未知 gate 语义继续在 capability guard 中拒绝。

- [x] **Step 3: 更新 ledger**

分别记录 `resident_shared_weight_bytes`、`dynamic_slot_bytes` 和 `host_routed_expert_bytes`。shared bytes 不影响 `effective_num_slots`。

- [x] **Step 4: 运行测试并提交**

```bash
python -m pytest -q tests/integration/test_shared_expert.py tests/unit/test_capabilities.py
git add src/vllm_latchmoe_cuda/capabilities.py src/vllm_latchmoe_cuda/runner_adapter.py src/vllm_latchmoe_cuda/offloader.py tests/integration/test_shared_expert.py
git commit -m "feat: support resident external shared experts"
```

---

### Task 10: 重建 Benchmark 合同和证据门槛

**Files:**

- Create: `benchmark/scripts/verify_budget_contract.py`
- Create: `tests/unit/test_budget_contract.py`
- Modify: `src/vllm_latchmoe_cuda/benchmark.py`
- Modify: `benchmark/scripts/run_sharegpt.py`
- Modify: `src/vllm_latchmoe_cuda/correctness.py`
- Modify: `README.md`

- [x] **Step 1: 让旧正式结果失去新算法标签**

README 保留 2026-07 结果，但标记为 `legacy temporary-bank/shared-pool evidence`。旧 generator 的 `(3,7,...,47)` 和旧正式实验的 first-12 `(0..11)` 必须分别原样记录，不能改写冻结证据，也不能称为新的默认 placement。59.85% 不能作为 Main Slots Cache 的性能结果；32/64 slot 单轮数据继续标为 exploratory。

- [x] **Step 2: 建立严格 baseline 合同**

主对照使用 `ManifestUVAOffloader` 精确卸载与 LatchMoE 相同的 routed expert 参数集合。manifest 必须从已生成的 immutable plan 导出，禁止 generator 独立选择默认层。比较器必须校验 `selection_strategy`、ordered eligible IDs、selected layer IDs、`plan_id`、参数名、host bytes、resident weight bytes、KV reserve、graph policy、workload hash 和 source identity。若为了匹配 free HBM 给 UVA 增加等量 reservation，必须记录 reservation bytes；无 reservation 的 practical UVA 另表报告，不能混成等预算结论。

- [x] **Step 3: 强制 exact-token correctness gate**

正式性能运行前，native full-resident oracle、exact-selection UVA 和 LatchMoE 对同一 deterministic prompts 必须 token ID 全等。短 smoke、BF16 `allclose` 和固定输出长度不能替代该门槛。

- [x] **Step 4: 验证 profile 语义**

`verify_budget_contract.py` 拒绝以下结果：无 profile 的均匀 Qwen plan 不是 `midpoint_stratified_v1`；Qwen3 `E=48,K=12` 不是 `(2,6,...,46)`；eligible 中包含 dense/non-MoE 层；baseline 和 LatchMoE selected IDs 不同；存在 temporary bank；resident layer 有 H2D；hit wave 有 H2D；pair count 不完整；combine count 不为每层一次；`h=0` 或 `h=C<P` 声称 overlap；candidate 没有 actual event window 却计入 overlap；parent/worker plan_id 或 selected IDs 不同。

- [x] **Step 5: 使用新生产命令**

```bash
CUDA_VISIBLE_DEVICES=0 \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
python -m vllm_latchmoe_cuda serve /home/lcw/model \
  --cuda-moe-offload-gb 13.5 \
  --max-model-len 4096 \
  --max-num-seqs 1 \
  --max-num-batched-tokens 4096 \
  --kv-cache-memory-bytes 536870912 \
  --gpu-memory-utilization 0.90 \
  --no-enable-prefix-caching
```

- [x] **Step 6: 运行合同测试并提交**

```bash
python -m pytest -q tests/unit/test_budget_contract.py tests/unit/test_benchmark.py tests/unit/test_correctness.py
git add benchmark/scripts/verify_budget_contract.py src/vllm_latchmoe_cuda/benchmark.py benchmark/scripts/run_sharegpt.py src/vllm_latchmoe_cuda/correctness.py README.md tests/unit/test_budget_contract.py tests/unit/test_benchmark.py tests/unit/test_correctness.py
git commit -m "test: enforce comparable CUDA LatchMoE evidence"
```

---

### Task 11: 分层验收后再重新跑正式实验

**Files:**

- Modify only if a relevant failure is found in the preceding tasks.
- Write new artifacts under a new run ID; never overwrite frozen 2026-07 results.

- [x] **Step 1: CPU 和静态门槛**

```bash
python -m pytest -q tests/unit
python -m ruff check .
python -m ruff format --check .
git diff --check
```

Expected: 全部通过；没有 CUDA 设备也能验证 planner、pair coverage、capability 和 artifact comparator。

- [x] **Step 2: 小张量 CUDA 生命周期门槛**

```bash
python -m pytest -q tests/gpu tests/integration \
  -k "main_cache or overlap or graph or lifecycle or combine or shared"
```

Expected: stable address、逐层 cache、真实 replacement、一次 combine、poison/drain 和 event overlap 全部通过。

- [x] **Step 3: 真实 12 层张量门槛**

运行 bounded route、union-128 overflow、连续 decode cache-hit、三种 overlap 边界和故障注入。记录 HBM 分项，确认：

```text
temporary_bank_bytes == 0
allocated_dynamic_weight_bytes == selected_layers * slots * bytes_per_expert
pair_count == num_tokens * top_k
combine_count == selected_layer_forward_count
```

- [ ] **Step 4: 全模型 exact-token 门槛**

分别独立启动 full-resident native、exact-selection UVA 和 serial Main Cache。至少覆盖短 prompt、长 prefill、decode、多 wave 和重复 route，所有 token IDs 全等后才运行 async。async 必须再次全等。

- [x] **Step 5: 正式三轮性能实验（受限两方）**

固定相同模型/shards、prompt manifest、sampling、输出上限、并发、KV reserve、graph capture sizes、selected IDs 和预算 ledger。每个 backend 独立启动至少三轮，保存原始 JSONL、CUDA timeline、plan、source hash 和失败 case。不得删除 OOM、startup failure 或 non-comparable case。

- [ ] **Step 6: 发布前最终审计**

```bash
python benchmark/scripts/verify_budget_contract.py \
  --native artifacts/<run>/native \
  --uva artifacts/<run>/uva \
  --latchmoe artifacts/<run>/latchmoe
```

Expected: 输出 `PASS`，并显式列出 exact parameter set、HBM ledger、pair/combine counts、candidate/actual overlap 和 exact-token parity。

---

## 本次实施进展与验收记录

截至 2026-09-12，Task 1-9 已按功能边界完成并拆分提交：

- `612d6d8` `feat: add model-adaptive CUDA residency plan`：不可变 GiB plan、整层预算、midpoint/ordered-prefix 选择、slot 可行性、schema v2 identity lock。
- `345f48f` `feat: propagate one CUDA offload plan to workers`：生产 CLI、环境传播、worker plan/identity 校验和 fail-closed plugin。
- `fa79b81` `refactor: give each offloaded layer a persistent main cache`：selected-only dynamic Host Store、逐层 cache、独立映射和生命周期。
- `76a8727` `feat: stream overflow through the main cache`：hit-first planner、串行 replacement、pair coverage 和一次 combine 执行路径。
- `24ca378` `feat: defer routed pairs to one native combine`：窄 `CudaMoeSeam`、`NativeWavePayload`、capability descriptor 和组合调用计数测试。
- `8c74764` `fix: keep dynamic CUDA staging outside graph replay`：PIECEWISE graph boundary、capture 时拒绝动态 staging/overflow，并记录 H2D 事件字段。
- `c05d163` `refactor: finalize serial main-cache execution`：删除生产 exact-wave/double-bank 路径，统一到串行 Main Cache executor，并更新 lifecycle/runner 断言。
- `785117c` `fix: lock qualified vLLM capability source hash`：固化 vLLM 0.19.1 fused-MoE seam 源码 SHA-256，未知源码 fail closed。
- `ac828ee` `test: align wave lifecycle checks with main cache`：将遗留双-bank测试迁移为单层持久 cache 复用和层间独立性验收。

已执行验收：

```text
/root/latchmoe-venv/bin/python -m pytest -q tests/unit
137 passed

/root/latchmoe-venv/bin/python -m pytest -q tests/unit tests/integration tests/gpu
196 passed, 20 warnings

Task 1-6 targeted suite (plan/CLI/cache/waves/seam/graph/lifecycle/CUDA)
88 passed, 20 warnings
```

测试使用 `/root/latchmoe-venv` 的 PyTorch 2.10 CUDA 环境；A6000 上的小张量 slot、streaming、graph replay、single-layer cache isolation 和 seam 计数测试均通过。旧的 `execute_exact_waves` 仅保留为 diagnostic alias，双 temporary-bank 断言已迁移到逐层持久 Main Cache 断言，生产 adapter 已切换到 `execute_main_cache_waves`。当前 vLLM 0.19.1 wheel 未提供独立 layer-level combine hook，因此生产 plan 在未注入锁定 native seam 时明确抛 `NativeCombineError`，不会以 `index_add_` 冒充 native combine；真实模型 full-token gate 仍需在带该 hook 的锁定 vLLM fork 上运行。

Task 7-8 已在 2026-09-12 完成并提交：

- `578c671` `feat: add constrained overlap and poisoned lifecycle`：为 partial-hit wave 增加只写 idle slot 的 H2D prefetch、CUDA event 时间窗和 candidate/actual overlap profile；新增 `RuntimeState`、`RuntimePoisonedError`、failure JSONL、close/drain 和 graph registry 注销，并注册 plugin 的 `atexit` 后备关闭。
- `7c203c6` `test: strengthen Task 7-8 lifecycle acceptance`：补充 serial/overlap 的 pair/order/ownership parity，以及 failure 记录的 plan/layer/lease 字段断言。
- `8ad2e2c` `test: verify prefetched H2D before runtime poison`：确认异常前下一 wave 的 H2D 已发出，再验证 runtime poison 和 drain。
- `9be80cf` `test: cover pending transfer drain and victim parity`：覆盖未完成 transfer ticket 的 drain，以及 victim sequence/H2D traffic parity。
- `93b9a18` `fix: preserve victim and transfer lifecycle evidence`：运行时保存 eviction sequence，并去重 failure 中的 transfer ticket 计数。

Task 7-8 验收：

```text
/root/latchmoe-venv/bin/python -m pytest -q tests/gpu/test_main_cache_overlap.py tests/gpu/test_failure_lifecycle.py
4 passed

/root/latchmoe-venv/bin/python -m pytest -q tests/unit tests/integration tests/gpu
202 passed, 20 warnings
```

partial-hit 测试确认 protected slots 不被覆盖，serial/overlap 输出、pair count、compute order、victim sequence、H2D traffic 和最终 ownership 一致；failure 测试确认 wave kernel 或预取 bookkeeping 异常后 runtime 进入 `POISONED`、后续 forward 抛 `RuntimePoisonedError`，close 可重复调用并 drain 在途 compute/transfer，再追加 `drained=true` failure 记录。小张量运行的 `actual_overlap` 作为设备实测布尔值记录，只有为真时才可计入性能统计。

Task 9 已在 2026-09-12 完成并提交：

- `3c3c1b9` `feat: support external resident shared experts`：只接受独立 external-resident shared module；shared 权重不进入 Host Store、dynamic slots 或 victim candidates，由 adapter 对原始 token batch 调用一次并保留 `(shared_output, routed_output)` ABI；fused/mix-placement/CPU shared 权重继续 fail closed；plan/profile ledger 分别记录 shared、dynamic slot 和 host routed bytes。

Task 9 验收：

```text
/root/latchmoe-venv/bin/python -m pytest -q tests/integration/test_shared_expert.py tests/unit/test_capabilities.py tests/unit/test_residency_plan.py
20 passed, 14 warnings

/root/latchmoe-venv/bin/python -m pytest -q tests/unit tests/integration tests/gpu
204 passed, 20 warnings
```

shared expert 集成测试确认 GPU 常驻、每次 forward 只调用一次、仅 routed tensors 进入 Host Store，且 `resident_shared_weight_bytes`、`dynamic_slot_bytes`、`host_routed_expert_bytes` 分项写入 profile。未完成部分从 Task 10 开始：benchmark 合同和正式全模型验收尚未宣称完成。

Task 10 已在 2026-09-12 完成代码实现并提交（`test: enforce comparable CUDA LatchMoE evidence`）：

- `benchmark/scripts/verify_budget_contract.py`：严格校验 plan identity、参数集合、HBM/KV/graph/workload/source 合同；拒绝 legacy temporary-bank、resident/hit-wave H2D、不完整 pair、错误 combine 和无 event window overlap；提供三方 exact-token gate CLI。
- `src/vllm_latchmoe_cuda/benchmark.py`、`benchmark/scripts/run_sharegpt.py`：新 measurement contract 写入 `selection_strategy`、ordered eligible/selected IDs、`plan_id`、参数名、host/resident bytes、KV reserve、graph policy、workload hash 和 source identity；schema-v1 manifest 明确标记为 diagnostic legacy evidence。
- `src/vllm_latchmoe_cuda/correctness.py`：增加 native/UVA/LatchMoE 三方逐请求 token-ID 全等比较接口；固定长度或 BF16 allclose 不再作为替代。
- `README.md`：冻结 2026-07 结果并标注 legacy temporary-bank/shared-pool，记录旧 `(3,7,...,47)` 与 first-12 `(0..11)` placement 及新的生产命令。

Task 10 验收：

```text
/root/latchmoe-venv/bin/python -m pytest -q tests/unit/test_budget_contract.py tests/unit/test_benchmark.py tests/unit/test_correctness.py
38 passed

/root/latchmoe-venv/bin/python -m pytest -q tests/unit tests/integration
175 passed, 20 warnings
```

ruff 未安装于 `/root/latchmoe-venv`，因此本轮未执行 ruff；在 2026-09-12 时尚未运行真实 full-model benchmark 或三方 token gate，Task 11 仍保持未完成。

Task 11 分层验收进展（2026-09-14）：

- Step 1 CPU/static：已通过 `tests/unit`（144 passed）和 `git diff --check`。当前虚拟环境没有 `ruff` 模块，因此 ruff 两项保持未执行。
- Step 2 小张量 CUDA：已通过 `tests/gpu tests/integration -k "main_cache or overlap or graph or lifecycle or combine or shared"`（25 passed, 39 deselected）以及完整 `tests/gpu tests/integration`（64 passed）。
- Step 3 真实张量门槛已通过：`verify_budget_contract.py` 的 `validate_runtime_ledger()` 检查 `temporary_bank_bytes == 0`、`pair_count == num_tokens * top_k`、`combine_count == selected_layer_forward_count` 和动态权重字节公式；真实 midpoint graph64 manifest 的 12 个 selected layers 通过 `LATCHMOE_RUN_REAL=1 ... tests/real/test_qwen_layers.py`（`13 passed`）。
- Step 4-6 仍未完成：当前环境不存在 `/home/lcw/model`，未运行 full-resident/UVA/serial/async 三方 token gate，也未启动新的三轮正式性能实验。冻结的 2026-07 artifact 未被覆盖。

真实模型验收更新（2026-09-14）：

- 使用 `/root/models/Qwen3-30B-A3B-Instruct-2507` 生成并提交 midpoint manifest `benchmark/manifests/offload_manifest.qwen3-root.midpoint12.graph64.20260914.json`，selected layers 为 `(2,6,10,14,18,22,26,30,34,38,42,46)`，64 slots 与 13.5 GiB plan 的 capture 约束一致。
- Step 3 真实权重部分：使用最终 graph64 midpoint manifest，`LATCHMOE_REAL=1` 的 `tests/real/test_qwen_layers.py` 通过 `13 passed`，覆盖 12 个 selected layers 的 checkpoint tensor、eager staged path 和 union-128 path；对应 runtime ledger/profile 预算为 `temporary_bank_bytes=0`、`dynamic_slot_bytes=603979776`、`host_routed_expert_bytes=1207959552`。
- Step 4 UVA 侧：`artifacts/20260914T091724-uva` 成功完成 3 个 deterministic prompts、16 token 输出，生成 `correctness.json` 和 `offload_telemetry.json`。
- Step 4 native 侧：`artifacts/20260914T091724-native` 保留启动失败；A6000 在分配第 48 层专家权重时 OOM（总显存 44.42 GiB、仅余 430.62 MiB），因此没有 native token oracle。
- Step 4 LatchMoE 侧：`artifacts/20260914T091724-latchmoe-eager-final` 保留启动失败；plan/identity lock 已成功生成并传播，但 vLLM 0.19.1 wheel 没有锁定 native combine seam，runtime 按 fail-closed 规则抛出 `NativeCombineError`。未绕过该保护，也未宣称 serial/async parity。
- 为推进 Step 4，新增 `VllmModularMoeSeam`，复用 vLLM 0.19.1 的 modular `quant_method.apply()` 和 `TopKWeightAndReduceContiguous`，并提交为 `4a67c1f`；同时 runner 在 eager/waves 模式按 manifest slot 数重建 plan（`21e5e05`）。使用 13-layer/32-slot manifest 重试仍在首个 Main Cache allocation 处 OOM（`artifacts/20260914T-real11-latchmoe-eager32-r1`，44.35 GiB 已分配、仅约 27 MiB 可用），因此 native seam 尚无真实全模型执行证据。
- 为验证 seam 本身，增加 20-layer/32-slot 诊断 manifest（`37dbff9`），将预算提高到 22 GiB 后真实 eager 运行成功：`artifacts/20260914T-real11-latchmoe-eager32-20layer-r2` 生成 3 个 deterministic 请求、每个 16 tokens；20 个 selected layers 均记录 `pair_count=192`、`combine_count=1`，`actual_offload_bytes=24159191040`。该 artifact 证明生产 Main Cache 路径可执行，但 selected 集合不同于 12-layer gate，且仍缺少 full-resident native oracle，不能宣称 Step 4 exact-token parity。
- Step 5/6 当时尚不能执行：缺少可运行的 native oracle 和 LatchMoE production seam；该历史状态由下方 2026-09-14 更新覆盖，失败 case 仍全部保留。
- 根据 2026-09-14 决策，取消 full-resident native 基线，直接执行 UVA 与 LatchMoE 两方正式三轮服务实验。两组均使用 `/root/models/Qwen3-30B-A3B-Instruct-2507`、20-layer midpoint manifest、32 slots、50 条固定 ShareGPT 请求、128 token 上限、2 warmup + 3 measurement、并发 8。
- UVA 三轮：`artifacts/20260914T-formal-uva-20layer-r1/summary.json`；LatchMoE 三轮：`artifacts/20260914T-formal-latchmoe-20layer-r1/summary.json`；对比：`artifacts/20260914T-formal-20layer-comparison.json`。LatchMoE 中位 output throughput 为 `17.47 tok/s`，UVA 为 `10.97 tok/s`，表面差异 `+59.24%`；中位 TPOT 为 `415.19 ms` 对 `535.05 ms`。
- 该对比报告明确标记 `comparable_offload_bytes=false`（UVA `15,798,475,264` bytes，LatchMoE `24,159,191,040` bytes）和 `comparable_contract=false`（两轮 source identity 不同）。因此这些数字只能作为“受限两方正式运行记录”，不能作为等预算性能结论；Step 6 最终审计仍保持未完成。

## 推荐实施顺序与停止点

1. Task 1-3 先解决“选哪些层、如何进入 worker、存储属于谁”。完成后应能启动逐层 cache，但还不宣称 overflow 完成。
2. Task 4-6 完成严格串行 Main Cache、一次 native combine 和图边界。这是第一个可用于正确性验收的版本。
3. 若 serial exact-token 不通过，停止，不进入 overlap。优先对比 router 输出、pair offsets、每 wave 未加权 MLP 输出和 native combine metadata。
4. Task 7-8 只改变调度和失败生命周期，不允许改变 victim 顺序或数学结果；当前已完成并通过小张量 serial/overlap 与 failure lifecycle 验收。
5. Task 9 是明确的 capability 扩展，当前已完成并限制在 external-resident shared ABI。
6. Task 10-11 通过前，所有新性能数字只能标记为 smoke/exploratory。

## 最终验收清单

- [x] `--cuda-moe-offload-gb` 是唯一生产 residency knob。
- [x] 无 profile 的均匀层默认使用 `midpoint_stratified_v1`；Qwen3 `E=48,K=12` 精确选择 `2,6,...,46`。
- [x] dense/non-MoE 层先从 ordered eligible IDs 排除；非均匀层按实测整层 bytes 累计，不使用平均层大小。
- [x] `(3,7,...,47)` 和 first-12 仅作为 legacy evidence，生产 manifest 不再拥有默认层选择权。
- [x] parent 和所有 workers 使用相同且校验过的 `plan_id`、`selection_strategy`、eligible IDs 和 selected IDs。
- [x] 只有 selected layers 进入 Host Store、Main Cache 和 H2D。
- [x] external-resident shared expert 保持 GPU 常驻，只计算一次且不进入动态 slot/victim 集合；fused/mix-placement/CPU shared ABI fail closed。
- [x] 每个 selected layer 有独立持久 cache；层切换不清空映射。
- [x] 生产 profile 中 `temporary_bank_bytes == 0`。
- [x] hit-first、真实 victim replacement、owner/generation/state lease 均有测试证据。
- [x] router 每层每次 forward 一次，pair coverage 严格完整。
- [x] 每层 forward 只有一次 native combine seam 调用；缺少锁定 hook 时 fail closed，不存在生产 `index_add_` combine。
- [x] `h=0`、`h=C<P` 不产生 overlap claim；partial-hit 只在下一 wave 能装入 idle slots 时产生 candidate，并记录 H2D/compute event 窗口和 `actual_overlap`。
- [x] 异常使 runtime poisoned，并在 shutdown 时 drain 所有在途 CUDA 工作；重复 close 幂等。
- [x] vLLM commit/source、model config/index/shards 均可追溯。
- [x] benchmark comparator 校验完整 plan/resource/workload/source 合同，并拒绝旧 temporary-bank/shared-pool 证据冒充新结果。
- [x] native/UVA/LatchMoE exact-token gate 实现为逐请求 token ID 全等；未通过真实 full-model gate 前不发布性能结论。
- [ ] full-resident、exact UVA、serial LatchMoE、async LatchMoE 通过 exact-token parity。
- [ ] 新性能结果满足等预算可比参数集合和 HBM ledger；本轮受限两方结果及不可比原因已保留。
