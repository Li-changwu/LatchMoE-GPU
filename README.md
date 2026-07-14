# LatchMoE CUDA

面向 **vLLM 0.19.1** 的独立 CUDA MoE offload 插件。当前实现将
LatchMoE/SEW-Offload 的可移植机制移植到 CUDA，并为
Qwen3-30B-A3B-Instruct-2507、BF16、TP=1 提供与受控 CUDA-UVA 基线共用
同一份 offload manifest 的验证路径。

本 README 是新服务器复现与正确性验证 runbook。设计、实现历史和原始证据索引见：

- [`docs/gpu_port/PROGRESS.md`](docs/gpu_port/PROGRESS.md)
- [`docs/superpowers/specs/2026-07-13-latchmoe-cuda-design.md`](docs/superpowers/specs/2026-07-13-latchmoe-cuda-design.md)
- [`docs/superpowers/plans/2026-07-13-latchmoe-cuda-implementation.md`](docs/superpowers/plans/2026-07-13-latchmoe-cuda-implementation.md)

## 当前状态与证据边界

已实现：

1. **Stage 1 - eager MVP**：目标 expert 直接加载到连续 pinned CPU HostStore，
   使用持久 CUDA slot tensor 和稳定 int32 `log2phy`/`expert_map`，同步 staging
   支持 active expert union 不超过 32。
2. **Stage 2 - async/lifecycle/graph**：独立 CUDA transfer stream/event，
   `EMPTY/LOADING/READY/COMPUTING` 状态保护，compute-done eviction guard，
   eager split 边界和 PIECEWISE CUDA Graph 支持。
3. **Stage 3 - exact waves**：active union 大于 32 时执行 capacity-bounded exact
   pair waves、共享双 stage buffer、事件保护和 scatter-add；每个 routed pair
   必须且只能计算一次。
4. **受控 UVA baseline**：UVA 和 LatchMoE 必须读取相同 manifest，选择完全相同的
   12 层 expert 参数。

源码服务器已经得到的证据：

- 完整回归：`123 passed, 13 skipped`；13 个 skip 是显式 gated 的 12 个真实
  Qwen layer case 和 1 个 full-model greedy case。
- 12 个 manifest layer 的真实权重数值对比全部通过；eager 最大绝对误差为
  `0.0`，exact-wave 最大绝对误差为 `0.014952659606933594`。
- eager/PIECEWISE synthetic graph smoke 均保持地址稳定且无 allocator 增长；该结果
  只覆盖 synthetic runtime，不是 full-model graph 结论。
- 源码服务器因外部进程占用约 39 GiB GPU 显存，full-model runs 在权重加载前被
  vLLM startup memory gate 拒绝。

尚未验证：

- UVA 与三个 LatchMoE mode 的 full-model greedy token 等价；
- full-model PIECEWISE capture；
- ShareGPT workload 下的重复性能测量；
- LatchMoE 相对 UVA 是提升、下降或仅在特定 workload 下提升。

**smoke、warmup、pytest 用时或单次 measurement 都不能作为最终性能结果。**

## 冻结实验合同

默认合同位于 [`benchmark/manifests/offload_manifest.json`](benchmark/manifests/offload_manifest.json)：

| 项目 | 固定值 |
| --- | --- |
| vLLM | `0.19.1` |
| vLLM tag commit | `b1388b1fbf5aaef47937fabe98931211684666a6` |
| 模型 | `Qwen3-30B-A3B-Instruct-2507` |
| 模型 revision | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| 默认模型路径 | `/root/models/Qwen3-30B-A3B-Instruct-2507` |
| dtype / TP | BF16 / TP=1 |
| expert 数 | 128 |
| slot 数 | 32 |
| offload layer | `3,7,11,15,19,23,27,31,35,39,43,47` |
| pinned HostStore | 13.5 GiB，低于 14 GiB 上限 |
| manifest SHA-256 | `55fb0e810af54f463fb8f0844698e30946d12dbedb1b476cdf01ca9633a0ed0b` |

所有 UVA/LatchMoE 对比必须使用同一份 manifest 和 manifest SHA-256。

## 参考环境

以下是源码服务器实际验证过的环境，不是根据兼容性推测出的版本：

| 组件 | 参考值 |
| --- | --- |
| OS / architecture | Linux x86_64，glibc 2.35 |
| GPU | NVIDIA RTX A6000，46068 MiB |
| NVIDIA driver | `580.159.03` |
| 主机内存 | 503 GiB |
| Conda | `26.1.1`，安装于 `/opt/miniconda3` |
| Python | `3.13.12` |
| PyTorch | `2.10.0+cu128` |
| torch CUDA runtime | `12.8` |
| vLLM | `0.19.1` |
| Triton | `3.6.0` |
| transformers | `5.10.2` |
| safetensors | `0.7.0` |
| pytest | `9.0.2` |
| Ruff | `0.15.21` |

当前只承诺 vLLM **严格等于 0.19.1**。不同 GPU、driver、Python、torch 或 Triton
组合必须作为新的环境重新验证，不能直接继承上述结果。

## 1. 获取代码

前置条件：新服务器需要安装 Git，并配置一个对私有仓库
`Li-changwu/LatchMoE-GPU` 有读取权限的 GitHub SSH key。先验证远端访问：

```bash
git ls-remote \
  git@github.com:Li-changwu/LatchMoE-GPU.git \
  refs/heads/cuda-latchmoe
```

该命令必须输出 `refs/heads/cuda-latchmoe`。没有 SSH key 时，应先为服务器配置 deploy
key 或用户 key；也可以使用带权限的 GitHub HTTPS credential，但不要把 token 写进
README、shell history 或 artifact。

```bash
git clone -b cuda-latchmoe \
  git@github.com:Li-changwu/LatchMoE-GPU.git
cd LatchMoE-GPU

# 运行时代码基线必须包含这个提交；后续 README-only 提交可以位于其后。
git merge-base --is-ancestor \
  205caead34b5c27f24ae32fdbdfcd5e18936014a HEAD

# 文档提交可以位于实现提交之后，但 runtime/test tree 必须与审查基线一致。
git diff --exit-code \
  205caead34b5c27f24ae32fdbdfcd5e18936014a -- \
  pyproject.toml benchmark scripts src tests

# 新 clone 必须干净；任何输出都应先归因。
test -z "$(git status --porcelain=v1 --untracked-files=all)"
```

上述命令退出 0 表示当前分支包含经过审查的三阶段实现，并且 runtime/test tree 未被
后续文档提交或本地修改改变。执行实验时保持 worktree 清洁；artifact manifest 会记录
commit、dirty diff 和未跟踪源码 SHA-256。

## 2. 创建 Python 环境并安装插件

前置条件：服务器已经在 `/opt/miniconda3` 安装 Miniconda，并能访问 Python package
index。推荐创建独立环境。以下命令先从 PyTorch 官方 CUDA 12.8 index 安装 torch，再
安装唯一目标 vLLM 及其宽松依赖的精确参考版本，最后以 `--no-deps` 安装插件，避免
editable install 改写已有 torch/vLLM：

```bash
/opt/miniconda3/bin/conda --version
source /opt/miniconda3/etc/profile.d/conda.sh
conda create -n latchmoe-cuda python=3.13.12 -y
conda activate latchmoe-cuda

python -m pip install --upgrade pip
python -m pip install \
  --index-url https://download.pytorch.org/whl/cu128 \
  "torch==2.10.0"
python -m pip install \
  "vllm==0.19.1" \
  "transformers==5.10.2" \
  "safetensors==0.7.0" \
  "pytest==9.0.2" \
  "ruff==0.15.21"
python -m pip install -e . --no-deps
```

vLLM 0.19.1 在 Linux x86_64 上固定依赖 torch 2.10.0，torch 2.10.0 固定依赖
Triton 3.6.0 和 CUDA 12.8 runtime package。`torch.__version__` 是否带 `+cu128` 仍取决于
所选 wheel/index。如果解析结果与下方检查不一致，先不要运行最终实验；记录实际版本，
并将其视为新的环境变量。

执行 fail-closed 环境检查：

```bash
python - <<'PY'
from importlib.metadata import entry_points, version
import torch

expected = {
    "vllm": "0.19.1",
    "triton": "3.6.0",
    "transformers": "5.10.2",
    "safetensors": "0.7.0",
    "pytest": "9.0.2",
    "ruff": "0.15.21",
}
actual = {name: version(name) for name in expected}
for name, wanted in expected.items():
    assert actual[name] == wanted, (name, actual[name], wanted)

assert torch.__version__ == "2.10.0+cu128", torch.__version__
assert torch.version.cuda == "12.8", torch.version.cuda
assert torch.cuda.is_available(), "CUDA is unavailable"

plugins = {
    entry.name: entry.value
    for entry in entry_points(group="vllm.general_plugins")
}
assert plugins.get("latchmoe_cuda") == "vllm_latchmoe_cuda.plugin:register", plugins

print({"packages": actual, "torch": torch.__version__, "cuda": torch.version.cuda})
print({"gpu": torch.cuda.get_device_name(0), "plugins": plugins})
PY
```

## 3. 准备模型

精确复现时，将模型放在 manifest 记录的绝对路径：

```text
/root/models/Qwen3-30B-A3B-Instruct-2507
```

如果新服务器还没有 checkpoint，可使用 vLLM 安装的 `huggingface_hub` 下载固定
revision。模型约 57 GiB，运行前确认磁盘空间：

```bash
mkdir -p /root/models
df -h /root/models

python - <<'PY'
from huggingface_hub import snapshot_download

path = snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Instruct-2507",
    revision="0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe",
    local_dir="/root/models/Qwen3-30B-A3B-Instruct-2507",
)
print(path)
PY
```

至少应存在：

```bash
test -f /root/models/Qwen3-30B-A3B-Instruct-2507/config.json
test -f /root/models/Qwen3-30B-A3B-Instruct-2507/model.safetensors.index.json
find /root/models/Qwen3-30B-A3B-Instruct-2507 \
  -maxdepth 1 -name 'model-*.safetensors' -type f | sort
```

校验冻结 manifest、模型 config 和 weight index：

```bash
python benchmark/scripts/generate_offload_manifest.py --check

python - <<'PY'
import json
from pathlib import Path
from vllm_latchmoe_cuda.manifest import OffloadManifest

path = Path("benchmark/manifests/offload_manifest.json")
manifest = OffloadManifest.load(path)
manifest.validate_model_files()
model_root = Path(manifest.model.path)
index = json.loads((model_root / "model.safetensors.index.json").read_text())
shards = sorted(set(index["weight_map"].values()))
missing = [name for name in shards if not (model_root / name).is_file()]
assert not missing, f"missing model shards: {missing}"
assert manifest.manifest_sha256 == (
    "55fb0e810af54f463fb8f0844698e30946d12dbedb1b476cdf01ca9633a0ed0b"
)
print(manifest.manifest_sha256)
print(manifest.model.path)
print(f"model shards: {len(shards)}")
PY
```

如果模型必须放在其他绝对路径，可生成新的 manifest：

```bash
python benchmark/scripts/generate_offload_manifest.py \
  --model /ABSOLUTE/PATH/TO/Qwen3-30B-A3B-Instruct-2507 \
  --output benchmark/manifests/offload_manifest.local.json
```

这会产生新的 canonical manifest SHA-256，不再是原 manifest 的字节级复现。该实验中
每个 UVA/LatchMoE runner 都必须显式传入同一个
`--manifest benchmark/manifests/offload_manifest.local.json`。真实 layer pytest 当前读取
默认 manifest，因此最稳妥的复现方式仍是使用默认模型路径。如果确实生成新 manifest，
应在独立实验分支提交该文件并恢复 clean worktree，再设置新的 `SOURCE_COMMIT` 和运行
artifact；不要从 dirty source 生成最终证据。后续命令和 manifest hash 断言均以默认
冻结 manifest 为准，使用新 manifest 时必须同步调整并记录这些断言。

## 4. 确认 GPU 空闲

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,driver_version \
  --format=csv,noheader,nounits
```

full-model runner 使用 `--gpu-memory-utilization 0.98`。vLLM 0.19.1 的启动门要求：

```text
startup_free_memory >= total_memory * gpu_memory_utilization
```

在 RTX A6000 上约等于启动时至少空闲 43.53 GiB。该检查发生在模型构造和 13.5 GiB
expert offload 之前。若 GPU 已被其他进程占用，应释放占用后再运行；不要把 utilization
降到很小来掩盖环境问题，因为越过启动门不代表模型随后能够装入。

## 5. 运行完整回归

为本次验证设置唯一 ID：

```bash
test -z "$(git status --porcelain=v1 --untracked-files=all)"
export RUN_ID="$(date -u +%Y%m%dT%H%M%S%N)-$(hostname -s)-$$"
export SOURCE_COMMIT="$(git rev-parse HEAD)"
mkdir -p artifacts

for suffix in \
  regression qwen-layers graph-eager graph-piecewise qwen-uva \
  qwen-latchmoe-eager qwen-latchmoe-piecewise qwen-latchmoe-waves
do
  test ! -e "artifacts/${RUN_ID}-${suffix}"
done

# pytest 目录由 shell 创建；runner 目录必须保持不存在，由 ArtifactRun 原子创建。
mkdir "artifacts/${RUN_ID}-regression"
```

执行所有测试和静态检查：

```bash
python -m pytest tests -q \
  --junitxml="artifacts/${RUN_ID}-regression/pytest.xml"
python -m ruff check .
python -m ruff format --check .
python benchmark/scripts/generate_offload_manifest.py --check
git diff --check
```

源码参考环境的结果是 `123 passed, 13 skipped`。此处的通过条件是命令退出 0；skip
必须仅来自：

- 12 个 `LATCHMOE_RUN_REAL=1` 真实 layer gate；
- 1 个 `LATCHMOE_RUN_E2E=1` full-model greedy gate。

该步骤证明实现回归通过，不提供吞吐或延迟结论。

## 6. 运行 12 层真实 Qwen 数值对比

```bash
mkdir "artifacts/${RUN_ID}-qwen-layers"

LATCHMOE_RUN_REAL=1 python -m pytest \
  tests/real/test_qwen_layers.py -q \
  --junitxml="artifacts/${RUN_ID}-qwen-layers/pytest.xml" \
  --basetemp="artifacts/${RUN_ID}-qwen-layers/tmp"
```

参考结果为 13 passed：1 个 checkpoint-index coverage case 加 12 个 layer case。每个
layer case 同时检查 staged eager 和 union-128 exact waves。pytest 用时不能作为性能数据。

## 7. 运行 synthetic CUDA Graph smoke

artifact runner 要求目标目录尚不存在：

```bash
python scripts/run_graph_checks.py \
  --mode eager \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-graph-eager"

python scripts/run_graph_checks.py \
  --mode piecewise \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-graph-piecewise"
```

检查以下字段：

```bash
python - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
for mode in ("eager", "piecewise"):
    path = Path(f"artifacts/{run_id}-graph-{mode}/graph_checks.json")
    report = json.loads(path.read_text())
    assert report["scope"] == "synthetic-runtime"
    assert report["addresses_stable"] is True
    assert report["staging_during_capture_rejected"] is True
    assert report["output_close"] is True
    assert report["allocated_growth_bytes"] <= report["allocated_growth_limit_bytes"]
    assert report["reserved_growth_bytes"] <= report["reserved_growth_limit_bytes"]
    if mode == "piecewise":
        assert report["captured_segment_count"] >= 1
    print(mode, report)
PY
```

这是 synthetic-runtime smoke，不证明 full-model vLLM PIECEWISE capture。

## 8. 运行 full-model greedy correctness

四个 mode 必须使用相同 manifest、prompts、sampling、BF16 和 TP=1。先运行 UVA：

```bash
python scripts/run_correctness.py \
  --mode uva \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-qwen-uva" \
  --max-tokens 16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory-bytes 268435456
```

UVA 成功后才运行三个 candidate。每个 candidate 直接读取同一次 UVA 的
`correctness.json`。comparator 自动检查 backend/graph policy、manifest SHA-256、模型
revision、dtype、TP、sampling、prompt、prompt token、output token 和 decoded text；这些
字段不一致会令命令非零退出：

```bash
python scripts/run_correctness.py \
  --mode latchmoe-eager \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-qwen-latchmoe-eager" \
  --reference-json "artifacts/${RUN_ID}-qwen-uva/correctness.json" \
  --max-tokens 16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory-bytes 268435456

python scripts/run_correctness.py \
  --mode latchmoe-piecewise \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-qwen-latchmoe-piecewise" \
  --reference-json "artifacts/${RUN_ID}-qwen-uva/correctness.json" \
  --max-tokens 16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory-bytes 268435456

python scripts/run_correctness.py \
  --mode latchmoe-waves \
  --kind smoke \
  --artifact-dir "artifacts/${RUN_ID}-qwen-latchmoe-waves" \
  --reference-json "artifacts/${RUN_ID}-qwen-uva/correctness.json" \
  --max-tokens 16 \
  --max-model-len 512 \
  --gpu-memory-utilization 0.98 \
  --kv-cache-memory-bytes 268435456
```

`latchmoe-waves` 还要求 profile 中出现真实 `exact_waves` event，否则即使进程生成了
文本也会失败。

`max_model_len`、`gpu_memory_utilization` 和 `kv_cache_memory_bytes` 当前没有写入
`correctness.json`，因此 comparator 不会自动比较它们。本 README 为四个 mode 固定了
相同值，第 9 节还会从每个 `run_manifest.json` 复核这些参数；不要省略该检查。

可再通过 pytest 独立检查任一 candidate artifact：

```bash
LATCHMOE_RUN_E2E=1 \
LATCHMOE_E2E_REFERENCE_JSON="artifacts/${RUN_ID}-qwen-uva/correctness.json" \
LATCHMOE_E2E_CANDIDATE_JSON="artifacts/${RUN_ID}-qwen-latchmoe-piecewise/correctness.json" \
python -m pytest tests/real/test_qwen_greedy.py -q
```

只有上述命令真实退出 0，才能写“该 candidate 与 UVA 在此 correctness workload 上
greedy token 完全一致”。它仍然不是性能结论。

## 9. 检查 artifact

每个 runner artifact 目录包含以下文件的全部或子集：

| 文件 | 含义 |
| --- | --- |
| `run_manifest.json` | kind、命令、commit、dirty source、状态、退出码 |
| `contract.json` | manifest/model/vLLM/dtype/TP/mode 合同 |
| `correctness.json` | prompt token、greedy output token 和文本 |
| `comparison.json` | candidate 与 UVA 的合同和 token 比较 |
| `graph_checks.json` | synthetic graph/address/memory 检查 |
| `profile.jsonl` | staging、wave 等运行事件 |
| `stdout.log` / `stderr.log` | 原始进程输出 |
| `failure.json` | 结构化失败原因与 traceback |
| `artifact_inventory.json` | artifact 文件 SHA-256 映射 |
| `SHA256SUMS` | 可由 `sha256sum -c` 验证的清单 |

先检查四个 full-model run 使用同一 source、manifest 和 runtime knobs：

```bash
python - <<PY
import json
from pathlib import Path

run_id = "${RUN_ID}"
source_commit = "${SOURCE_COMMIT}"
names = (
    "qwen-uva",
    "qwen-latchmoe-eager",
    "qwen-latchmoe-piecewise",
    "qwen-latchmoe-waves",
)
manifest_hashes = set()
for name in names:
    root = Path(f"artifacts/{run_id}-{name}")
    run = json.loads((root / "run_manifest.json").read_text())
    contract = json.loads((root / "contract.json").read_text())
    command = run["command"]

    def option(flag):
        return command[command.index(flag) + 1]

    assert run["status"] == "completed", (name, run)
    assert run["exit_code"] == 0, (name, run)
    assert run["git_commit"] == source_commit, (name, run["git_commit"])
    assert run["git_dirty"] is False, (name, run["git_status"])
    assert option("--max-tokens") == "16"
    assert option("--max-model-len") == "512"
    assert option("--gpu-memory-utilization") == "0.98"
    assert option("--kv-cache-memory-bytes") == "268435456"
    manifest_hashes.add(contract["manifest_sha256"])

assert manifest_hashes == {
    "55fb0e810af54f463fb8f0844698e30946d12dbedb1b476cdf01ca9633a0ed0b"
}
print({"source_commit": source_commit, "manifest_hashes": manifest_hashes})
PY
```

然后检查 PIECEWISE correctness artifact 的 inventory 和 token comparison：

```bash
ARTIFACT="artifacts/${RUN_ID}-qwen-latchmoe-piecewise"

(cd "$ARTIFACT" && sha256sum -c SHA256SUMS)

python - <<PY
import json
from pathlib import Path

root = Path("${ARTIFACT}")
run = json.loads((root / "run_manifest.json").read_text())
comparison = json.loads((root / "comparison.json").read_text())
assert run["schema_version"] == 1
assert run["status"] == "completed"
assert run["exit_code"] == 0
assert run["kind"] == "smoke"
assert run["final_result"] is False
assert comparison["match"] is True
print(run)
print(comparison)
PY
```

失败 artifact 不应删除或改写。保留 `status=failed`、`failure.json`、日志和 SHA 清单，
根据原始 traceback 归因。

## 10. 常见失败

### vLLM startup memory gate

```text
Free memory ... is less than desired GPU memory utilization
```

原因是启动空闲显存低于 `total_memory * gpu_memory_utilization`。这发生在 offload 前；
释放其他进程的 GPU 占用后使用新的 `RUN_ID` 重跑。

### vLLM 版本不匹配

插件只支持 0.19.1。不要绕过版本守卫；建立正确环境后重跑。

### manifest stale 或模型校验失败

确认模型绝对路径、`config.json`、`model.safetensors.index.json` 与冻结合同一致。若模型
路径不同，按第 3 节生成新的 manifest，并确保所有对比 mode 使用同一个新文件。

### artifact 目录已存在

runner 默认拒绝覆盖已有目录。设置新的 `RUN_ID`，不要删除旧证据后复用名字。

### `exact_waves` event 缺失

waves mode 只有真实走过 exact-wave 路径才算成功。检查 `profile.jsonl` 和 routing active
union，不要把普通 eager 成功当成 waves 成功。

### token mismatch

检查 `comparison.json` 中的 `contract_errors`、`mismatched_requests` 和
`mismatched_text_requests`，同时保留 UVA/candidate 的两个 `correctness.json`。

## 性能实验边界

当前仓库提供 correctness runner、synthetic graph runner 和 final artifact promotion
完整性检查，但尚未提供：

- ShareGPT 数据集定位/预处理；
- 固定 request-rate、prefill/decode mix 和并发矩阵；
- UVA/LatchMoE warmup 与重复 measurement orchestration；
- profiler/吞吐/延迟/显存结果的最终统计汇总。

因此，本 README 可验证三阶段实现和 full-model correctness，不能单独回答 LatchMoE
相对 UVA 的性能提升或下降。最终 claim 必须来自同一 manifest、同一 workload、至少三次
有效 measurement 的原始 JSON、日志和 profiler artifact，并由研究负责人确认。

## 目录结构

```text
benchmark/manifests/             冻结 offload manifest
benchmark/scripts/               manifest 生成与检查
scripts/run_correctness.py       full-model greedy artifact runner
scripts/run_graph_checks.py      synthetic eager/PIECEWISE graph runner
src/vllm_latchmoe_cuda/          插件、HostStore、runtime、waves、UVA adapter
tests/unit/                      CPU/portable contract tests
tests/integration/               vLLM 生命周期与 adapter tests
tests/gpu/                       CUDA layout/lifecycle/numerics/graph tests
tests/real/                      opt-in Qwen layer 与 full-model artifact tests
docs/gpu_port/PROGRESS.md        命令、artifact、失败与下一步记录
```

## License

Apache-2.0。可移植逻辑的来源与适配说明保留在相关源码文件头中。
