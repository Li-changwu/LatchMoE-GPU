# LatchMoE CUDA

LatchMoE/SEW-Offload 在 CUDA + vLLM 0.19.1 上的独立 MoE offload 插件。当前目标是
把 NPU LatchMoE 的 HostStore、有限 GPU expert slot、异步搬运、生命周期保护和 exact
pair waves 移植到 GPU，并与 vLLM 官方 CUDA UVA offloader 做同机、同模型、同负载对比。

设计与实现历史见 [`docs/gpu_port/PROGRESS.md`](docs/gpu_port/PROGRESS.md)。

## 实测结论

2026-07-15 在单卡 NVIDIA RTX A6000 48 GB 上完成了正式 eager 对比。模型为本机
`/home/lcw/model` 的 Qwen3-30B-A3B，ShareGPT 固定取 50 条，每条强制生成 128 token，
并发为 8。每个 backend 独立启动一次 server，先做 2 条 warmup，再连续做 3 轮
measurement；下表中的值是三轮统计量的中位数。

| 指标 | 官方 UVA | LatchMoE eager | 相对 UVA |
| --- | ---: | ---: | ---: |
| TTFT p50 | 7912.819 ms | 4541.042 ms | 降低 42.61% |
| TTFT p99 | 67985.256 ms | 35582.418 ms | 降低 47.66% |
| TPOT p50 | 504.165 ms/token | 244.752 ms/token | 降低 51.45% |
| TPOT p99 | 858.751 ms/token | 396.570 ms/token | 降低 53.82% |
| 输出吞吐 | 11.9639 token/s | 23.8696 token/s | 提高 99.51% |

指标由 vLLM 0.19.1 官方 `vllm bench serve` 计算。比较器对低延迟指标使用
`(UVA - LatchMoE) / UVA`，对吞吐使用 `(LatchMoE - UVA) / UVA`。

### 三轮原始值

| Backend / 轮次 | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms/token) | TPOT p99 (ms/token) | 输出吞吐 (token/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| UVA / 1 | 8482.381 | 70348.791 | 504.165 | 858.751 | 11.8830 |
| UVA / 2 | 7753.382 | 65598.260 | 517.315 | 863.793 | 11.9639 |
| UVA / 3 | 7912.819 | 67985.256 | 499.073 | 827.669 | 12.0545 |
| LatchMoE / 1 | 4698.312 | 37694.233 | 244.752 | 396.570 | 23.6648 |
| LatchMoE / 2 | 3523.201 | 33762.247 | 258.502 | 378.561 | 23.9940 |
| LatchMoE / 3 | 4541.042 | 35582.418 | 243.907 | 401.222 | 23.8696 |

两侧都完成 50/50 请求且无失败；每轮输入 14,278 token、输出 6,400 token。

## 核心实现

1. 选中的 expert 权重直接加载到连续 pinned CPU HostStore，避免先在 GPU 构造完整
   expert tensor 再搬回 CPU。
2. 每层使用 32 个持久 CUDA slot、稳定的 `log2phy`/expert map 和受保护 LRU。
   transfer stream、CUDA event 和 `EMPTY/LOADING/READY/COMPUTING` 状态机避免加载、
   计算和淘汰之间的数据竞争。
3. active expert 不超过 32 时走常规 slot 路径；超过容量时按 routed pair 拆成
   capacity-bounded exact waves，使用共享双 stage bank。每个 routed pair 必须且只计算
   一次。
4. GPU planner 只把最多 128 个 active expert ID 同步到 CPU。pair offsets、logical /
   physical expert ID 和 routing weight 保留在 CUDA；同一层所有 wave 的结果合并为一次
   `index_add_`。
5. LatchMoE manifest 选择 0-11 层的全部 expert 参数。为保证容量口径相同，剩余
   offload budget 继续使用 vLLM 官方 UVA。正式 profile 记录了 3,394 个真实
   `exact_waves` event，均为 `pair_planner_mode=cuda_device`、
   `scatter_mode=layer_index_add`，实际出现 2-4 个 wave，单层最多 15,480 routed pairs。

官方基线的实现类为
`vllm.model_executor.offloader.uva.UVAOffloader`。两侧实际 offload 都是
`15,798,475,264` bytes：LatchMoE 由 `14,495,514,624` bytes manifest expert 权重和
`1,302,960,640` bytes residual UVA 组成。官方 UVA 按其原生参数顺序选择 offload
对象，LatchMoE 按 expert manifest 选择；本实验控制的是相同实际字节量，不是相同参数
集合。这正是两种 offload 策略的系统级对比边界。

## 冻结实验合同

正式 measurement 绑定到源提交
`63a7ed49f4fa1b83d307bf7e28c291af9a31be0d`，两个 backend 的
`workload_contract_sha256` 均为
`bcda4b8da72118cc86fec7dba75258697b492247f927c659a335afb4f10af65b`。

| 项目 | 固定值 |
| --- | --- |
| GPU | NVIDIA RTX A6000，46068 MiB，driver 580.159.03 |
| Python / PyTorch | 3.13.12 / 2.10.0+cu128 |
| CUDA / Triton / vLLM | 12.8 / 3.6.0 / 0.19.1 |
| 模型 | `/home/lcw/model`，Qwen3-30B-A3B |
| 模型 revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| dtype / TP | BF16 / 1 |
| manifest | `benchmark/manifests/offload_manifest.qwen3-base-ad44.first12.local.json` |
| manifest SHA-256 | `f03b0c416b8ed5181dfb3d7d7c85c1263d55a1acbc05d66336a50f8face673b8` |
| offload layer / slot | 0-11 / 每层 32 |
| ShareGPT | `/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json` |
| 数据集大小 / 记录数 | 672,837,942 bytes / 94,145 条，其中 92,886 条至少两轮 |
| 数据集 SHA-256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| 请求 | seed 42，50 条，request rate `inf`，concurrency 8 |
| 输出 | 每条 128 token，temperature 0，`ignore_eos` |
| 调度 | `max_num_seqs=8`，`max_model_len=2048`，`max_num_batched_tokens=2048` |
| KV / prefix cache | 256 MiB / 禁用 |
| 重复 | 每个 backend 2 warmup + 3 measurement |
| graph policy | UVA 和 LatchMoE 都为 `--enforce-eager` |

由于 `ignore_eos` 固定输出长度，本结果测的是同长度 serving 性能，不评价生成质量。

## 正确性边界

- 完整测试结果为 `150 passed, 1 skipped`。唯一 skip 是需要外部 UVA reference
  artifact 的严格 full-model greedy token 测试；12 个真实模型层数值测试均实际运行。
- 真实 vLLM Triton slot 路径的单层对比最大绝对误差为 `0.00390625`，平均绝对误差约
  `3.53e-4`，并通过 BF16 数值容差。
- 严格 full-model greedy token 曾与 UVA 不一致。checkpoint 权重抽样逐项匹配，真实
  Triton 层级数值测试通过，因此当前证据支持“BF16 offload 数值路径在层级容差内”，
  不支持“端到端 token 完全一致”。多层 BF16 路径的微小误差可能在 greedy 决策边界
  累积放大。
- 本次正式结果只比较 eager。synthetic PIECEWISE capture 已有 smoke 覆盖，但没有
  full-model PIECEWISE 性能数据，不能把上表推广为 PIECEWISE 结论。
- LatchMoE 测量期间显存峰值约 45.37 GiB，只剩约 117 MiB。当前合同在 A6000 上接近
  容量边界；driver、allocator、模型或并发变化都可能触发 OOM。

## 复现实验

使用已安装的环境：

```bash
cd /home/lcw/LatchMoE-GPU
PY=/home/lcw/miniconda3/envs/latchmoe-cuda/bin/python
MANIFEST=benchmark/manifests/offload_manifest.qwen3-base-ad44.first12.local.json
DATASET=/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

先校验两个本机 manifest，避免模型文件变化后继续使用旧合同：

```bash
$PY benchmark/scripts/generate_offload_manifest.py \
  --model /home/lcw/model \
  --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --output benchmark/manifests/offload_manifest.qwen3-base-ad44.local.json \
  --check

$PY benchmark/scripts/generate_offload_manifest.py \
  --model /home/lcw/model \
  --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --output "$MANIFEST" \
  --check
```

运行回归与静态检查：

```bash
LATCHMOE_REAL_MANIFEST="$MANIFEST" \
LATCHMOE_RUN_REAL=1 \
$PY -m pytest tests -q
$PY -m ruff check .
$PY -m ruff format --check .
git diff --check
```

`LATCHMOE_REAL_MANIFEST` 不能省略；测试套件的历史默认值仍指向旧的
`/root/models/Qwen3-30B-A3B-Instruct-2507` 合同。

正式 benchmark 目录不能已存在。使用新的 run id，顺序执行 UVA 和 LatchMoE；同一
时刻只能让一个 server 使用端口 8026：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%S)

$PY benchmark/scripts/run_sharegpt.py \
  --mode uva \
  --artifact-dir "artifacts/${RUN_ID}-uva" \
  --manifest "$MANIFEST" \
  --dataset "$DATASET" \
  --repetitions 3 --num-prompts 50 --output-len 128 \
  --max-concurrency 8 --max-num-seqs 8 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 268435456 --port 8026

$PY benchmark/scripts/run_sharegpt.py \
  --mode latchmoe-eager \
  --artifact-dir "artifacts/${RUN_ID}-latchmoe-eager" \
  --manifest "$MANIFEST" \
  --dataset "$DATASET" \
  --repetitions 3 --num-prompts 50 --output-len 128 \
  --max-concurrency 8 --max-num-seqs 8 \
  --max-model-len 2048 --max-num-batched-tokens 2048 \
  --kv-cache-memory-bytes 268435456 --port 8026

$PY benchmark/scripts/compare_sharegpt.py \
  --uva-summary "artifacts/${RUN_ID}-uva/summary.json" \
  --latchmoe-summary "artifacts/${RUN_ID}-latchmoe-eager/summary.json" \
  --output "artifacts/${RUN_ID}-comparison.json"
```

runner 会拒绝少于 3 轮、不是 50 条、合同不一致、请求失败或 offload 字节量不一致的
最终结果。它还会清除本地请求的 proxy 环境，避免 `127.0.0.1` 被 SOCKS proxy 拦截。

## 正式 artifacts

- `artifacts/sharegpt-c8-uva-20260715/`
- `artifacts/sharegpt-c8-latchmoe-eager-20260715/`
- `artifacts/sharegpt-c8-comparison-20260715.json`

两个正式目录的 `run_manifest.json` 都是 `status=completed`、`exit_code=0`、
`final_result=true`，各自的 `SHA256SUMS` 已通过 `sha256sum -c`。原始
`raw_result.json`、client/server log、每轮 normalized measurement、完整合同和
LatchMoE profile 均保留在目录内。

以下是失败证据，不应覆盖或删除：

- `artifacts/sharegpt-c1-uva-20260715/`：本地请求误走 proxy，50/50 失败。
- `artifacts/sharegpt-c1-uva-20260715-r2/`：concurrency 1 单轮耗时过长，人工中止，
  保留 `KeyboardInterrupt` artifact。

## 目录结构

```text
benchmark/manifests/             冻结 offload manifest
benchmark/scripts/               manifest、ShareGPT runner 和比较器
src/vllm_latchmoe_cuda/          插件、HostStore、runtime、waves、benchmark 合同
scripts/                         correctness 与 synthetic graph runner
tests/                           unit、integration、GPU 和真实模型测试
docs/gpu_port/PROGRESS.md        实现历史、实验记录与证据边界
artifacts/                       本机原始实验产物（Git 忽略）
```

## License

Apache-2.0。可移植逻辑的来源与适配说明保留在相关源码文件头中。
