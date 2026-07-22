# LatchMoE CUDA

LatchMoE/SEW-Offload 在 CUDA + vLLM 0.19.1 上的独立 MoE offload 插件。当前目标是
把 NPU LatchMoE 的 HostStore、有限 GPU expert slot、异步搬运、生命周期保护和 exact
pair waves 移植到 GPU，并与 vLLM 官方 CUDA UVA offloader 做同机、同模型、同负载对比。

设计与实现历史见 [`docs/gpu_port/PROGRESS.md`](docs/gpu_port/PROGRESS.md)。

## 图重放对图重放结论

2026-07-15 在单卡 NVIDIA RTX A6000 48 GB 上完成了正式 PIECEWISE CUDA Graph
对比。模型为 `/home/lcw/model` 的 Qwen3-30B-A3B，ShareGPT 固定取 50 条，每条强制
生成 128 token，并发为 8。每个 backend 独立启动一次 server，启动阶段预捕获
1/2/4/8-token 图，之后先做 2 条 warmup，再连续做 3 轮 measurement。下表取三轮
统计量的中位数。

| 指标 | 官方 UVA PIECEWISE | LatchMoE PIECEWISE | 相对 UVA |
| --- | ---: | ---: | ---: |
| TTFT p50 | 9942.701 ms | 4112.697 ms | 降低 58.64% |
| TTFT p99 | 83348.327 ms | 49741.911 ms | 降低 40.32% |
| TPOT p50 | 578.740 ms/token | 377.351 ms/token | 降低 34.80% |
| TPOT p99 | 969.359 ms/token | 633.805 ms/token | 降低 34.62% |
| 输出吞吐 | 10.1169 token/s | 16.1718 token/s | 提高 59.85% |

指标由 vLLM 0.19.1 官方 `vllm bench serve` 计算。比较器对延迟使用
`(UVA - LatchMoE) / UVA`，对吞吐使用 `(LatchMoE - UVA) / UVA`。两侧每轮均完成
50/50 请求且零失败，每轮输入 14,278 token、输出 6,400 token。

### 三轮原始值

| Backend / 轮次 | TTFT p50 (ms) | TTFT p99 (ms) | TPOT p50 (ms/token) | TPOT p99 (ms/token) | 输出吞吐 (token/s) |
| --- | ---: | ---: | ---: | ---: | ---: |
| UVA graph / 1 | 9942.701 | 90186.894 | 578.740 | 980.697 | 9.9876 |
| UVA graph / 2 | 9592.722 | 81067.206 | 594.548 | 927.957 | 10.1169 |
| UVA graph / 3 | 10274.072 | 83348.327 | 569.167 | 969.359 | 10.3244 |
| LatchMoE graph / 1 | 4276.263 | 46739.238 | 376.666 | 690.005 | 16.1718 |
| LatchMoE graph / 2 | 3846.913 | 52281.366 | 377.351 | 633.805 | 16.0259 |
| LatchMoE graph / 3 | 4112.697 | 49741.911 | 381.442 | 587.834 | 16.2434 |

## 为什么 UVA 有图仍然慢

CUDA Graph 解决的是 CPU dispatch 和 kernel launch 开销，不会改变权重所在的存储
层级。官方 `UVAOffloader` 把参数放在 pinned CPU memory，再建立稳定 CUDA view；稳定
指针使这些 kernel 可以 capture/replay，但每次 replay 时，MoE kernel 仍要通过 PCIe
按需读取 host 权重。

```text
UVA:      graph replay -> FusedMoE(host UVA pointer) -> PCIe demand loads
LatchMoE: active expert staging -> graph replay -> FusedMoE(stable HBM slots)
```

Qwen3-30B-A3B 的单个 expert 权重为 9 MiB，batch 8 的 decode step 有 64 个 routed
pairs。LatchMoE 先对 active expert 去重，再用连续 H2D copy 把每个命中的 expert 搬一次，
随后图内 Triton MoE 从 HBM 读取。UVA 则让计算 kernel 直接承担远端访问，无法把
`cudaMemcpyAsync` 的批量传输效率和 HBM 带宽带进计算阶段。因此两侧都 replay 后，
权重数据路径仍然不同，图启动开销也不是这个 workload 的主要瓶颈。

当前 graph 实现没有跨 token cache 命中，也没有把下一层权重预取隐藏在当前层计算后：
12 个 offload 层共享一个 slot pool，层切换会使上一层映射失效；stage 完成后当前层才
计算。所以本次保留下来的主要优势应归因于“去重后的连续 DMA + HBM 计算”，不能归因
于尚未实现的跨层 overlap。突发并发下更快的 decode 还会更快排空队列，因此 TTFT 的
改善同时包含了系统吞吐改善；TPOT 和输出吞吐更直接反映 decode 差异。

## 图边界与代价

- 两侧都是 `cudagraph_mode=PIECEWISE`，启动时都实际 capture 1/2/4/8-token 图，正式
  profile 都记录了 `cudagraph_capture` 和 `cudagraph_replay`，runtime mode 为
  `PIECEWISE`。
- 1-8 token 的 decode/mixed batch 使用图重放。512-token prefill chunk 和其他大于 8
  的动态 token shape 在日志中是 `Runtime Mode=NONE`。因此这是公平的 decode
  graph-to-graph 结果，不应描述成任意 prompt shape 的 full-model CUDA Graph。
- graph 模式用 128 个 identity slot：logical expert `e` 永远写到 slot `e`。12 层共享
  同一个 1.125 GiB pool，而不是每层各分配一份。稳定地址让图内 MoE kernel 可以重放，
  代价是 LatchMoE 比无 staging cache 的 UVA 多约 1.125 GiB device memory。
- 两侧实际 offload 都是 `15,798,475,264` bytes。LatchMoE 由
  `14,495,514,624` bytes manifest expert 权重和 `1,302,960,640` bytes residual
  official UVA 组成。控制的是相同实际字节量，不是完全相同的参数集合。
- 由于 BF16 多层误差和异步调度，两侧以及同一 backend 的不同轮次都不保证生成文本
  逐 token 相同。固定的是请求、输入 token 总数、每条输出长度和 shape，不是路由轨迹
  或生成质量；本结果是系统 serving 性能结论。

## 核心实现

1. 选中的 expert 权重直接加载到连续 pinned CPU HostStore，避免先在 GPU 构造完整
   expert tensor 再搬回 CPU。
2. `latchmoe::stage_experts` 和 `latchmoe::finish_experts` 是 PIECEWISE splitting
   boundary；动态 active-expert 解析、H2D 和生命周期管理留在图外。
3. `latchmoe::fused_moe_compute` 是 opaque custom op，位于捕获段内并调用 vLLM modular
   Triton MoE kernel。slot、map 和 graph token 的地址在整个 server 生命周期内稳定。
4. graph registry 使用稳定 `layer_id`，128 identity slot 消除了 logical/physical expert
   ID 的动态变化。共享 slot pool 通过 CUDA event 保证上一层计算结束后才能覆盖。
5. Latch adapter 显式推进 vLLM `forward_context.moe_layer_index`。缺少这一步会让第一个
   非 offload MoE 层错误解析为 layer 0，并使后续层全部错位。
6. eager 路径保留 LRU、CUDA device exact-pair planner 和双 stage bank；有限槽位 graph
   对捕获 shape 走单工作集 replay，对非捕获 overflow shape 走 exact waves。

## 2026-07-16 有限槽位 CUDA Graph 改进

本轮审计确认，旧 graph 路径虽然能 replay，但强制 `num_slots == num_experts`。这使
Qwen3 的 128 个 identity slot 占用约 1.125 GiB，也让真正的有限显存配置只能退回
eager。现在 PIECEWISE 支持有限 main slots，并以捕获容量合同保证图内不会 overflow：

```text
required_main_slots = min(num_experts, max_cudagraph_capture_size * top_k)
```

当前 `max_cudagraph_capture_size=8`、`top_k=8`，因此使用 64 个 main slots。更大的
非捕获 prefill/mixed shape 走 exact pair waves。wave capacity 与 main capacity 解耦，默认
为 `min(main_slots, 32)`，可用 `VLLM_LATCHMOE_WAVE_SLOTS` 调整。双缓冲 bank 0 复用
main pool，只有 bank 1 是额外分配；64 main + 32 wave 的总设备权重区为 96 个 expert
slot，约 864 MiB。

overflow 路径还改为让 vLLM modular Triton MoE kernel 直接读取 stage bank，删除旧实现
每个 wave 的 `stage bank -> main slot` 整槽 D2D copy。共享 main/stage bank 的覆盖由
CUDA event 连接到下一次 H2D，避免跨 stream 提前复用。

针对单并发的 1×8 路由，还增加了 active-set 自适应路径：不再为 8 个 ID 启动 GPU
`torch.unique`，而是直接 D2H 64 bytes 后在 CPU 去重；A6000 微基准由约 `58.8 us`
降至 `12.4 us`。超过 512 个 routed IDs 时仍使用 GPU unique。512-byte `log2phy`
发布也改为复用 event 守护的 pinned map buffers，避免每层重复创建 host tensor。

完整 Qwen smoke 使用 64-slot manifest，实际完成 1/2/4/8-token PIECEWISE 图捕获与
重放；同一运行的 192 routed-pair prefill 走了 3 个 32-slot exact waves。与官方 UVA
eager reference 的 3 个 deterministic prompt、12 个生成 token 完全一致，实际卸载
字节均为 `15,798,475,264`。

按“先测单并发”的顺序，另做了 50 请求、每请求 32 token、`max_concurrency=1`、
`max_num_seqs=1` 的单轮 exploratory graph-to-graph 检查。该配置使用 32 main slots 和
32-slot wave bank，总设备权重区为 64 个 expert slot（约 576 MiB），只捕获 1-token 图。
LatchMoE 为 `6.8072 token/s`，UVA 为 `4.6069 token/s`，前者高 `47.76%`；TPOT p50
由 `144.79` 降至 `115.42 ms/token`，TTFT p50 由 `2189.12` 降至 `968.12 ms`。两侧
workload hash、source state 和 offload bytes 相同。该结果只有一轮且输出长度不同于
上面的正式 128-token 合同，不能替代三轮正式结果；它也早于最后两项 host-control
微优化，因此是当前实现的保守 serving 基线。

详细设计审计与剩余风险见
[`docs/gpu_port/DESIGN_REVIEW_20260716.md`](docs/gpu_port/DESIGN_REVIEW_20260716.md)。

### Eager 历史结果

此前相同模型和 50 条数据的 eager 合同测得 UVA `11.9639 token/s`、LatchMoE
`23.8696 token/s`，吞吐提高 `99.51%`。但 eager 使用 32 slots 和
`max_num_batched_tokens=2048`，本次 graph 使用 128 identity slots 和 512-token 上限，
所以两组结果不能当作只改变 graph 开关的严格 ablation。

## 冻结实验合同

两个 graph backend 的 `workload_contract_sha256` 都是
`e90a0c2fb04f968432984e64ced7ea56994c3dda53b00051b8192eaf73ec27bc`，正式
artifact 绑定到相同 `source_state_sha256`：
`855bc37b916756aa1cba66eec0ff1323454feadab9fc369d7d90f63cad272786`。

| 项目 | 固定值 |
| --- | --- |
| branch / base commit | `cuda-latchmoe` / `02eb698369cd0b5cbe7c3caecec3987bf64293d0` |
| GPU | NVIDIA RTX A6000，46068 MiB，driver 580.159.03 |
| Python / PyTorch | 3.13.12 / 2.10.0+cu128 |
| CUDA / Triton / vLLM | 12.8 / 3.6.0 / 0.19.1 |
| 模型 | `/home/lcw/model`，Qwen3-30B-A3B，BF16，TP=1 |
| 模型 revision | `ad44e777bcd18fa416d9da3bd8f70d33ebb85d39` |
| manifest | `benchmark/manifests/offload_manifest.qwen3-base-ad44.first12.graph128.local.json` |
| manifest SHA-256 | `3c5a0c8fad70caf6e1c697d2a2471b798fc93dc3158da6bfddb8f8575f51c5e5` |
| offload layer / graph slot | 0-11 / 128 identity slots，共享 pool |
| ShareGPT | `/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json` |
| 数据集 SHA-256 | `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| 请求 | seed 42，50 条，request rate `inf`，concurrency 8 |
| 输出 | 每条 128 token，temperature 0，`ignore_eos` |
| 调度 | `max_num_seqs=8`，`max_model_len=2048`，`max_num_batched_tokens=512` |
| KV / prefix cache | 256 MiB / 禁用 |
| graph | PIECEWISE，capture sizes 1/2/4/8，compile cache 禁用 |
| 重复 | 每个 backend 2 warmup + 3 measurement |

2048-token compile range 在 A6000 的 graph LatchMoE 初始化阶段会 OOM，因此两侧统一
冻结为 512；不能把旧 eager 的 2048-token 合同与本表直接混用。

## 正确性边界

- 全量测试结果为 `161 passed, 1 skipped`。唯一 skip 是需要显式
  `LATCHMOE_RUN_E2E=1` 的完整 Qwen greedy 对比；12 个真实 checkpoint 层的
  32-active 和 128-active 数值路径均实际运行。
- full-model smoke 中，官方 UVA 和 LatchMoE PIECEWISE 对同一 deterministic prompt
  生成了相同短输出，且两份 profile 都包含真实 capture/replay。
- 真实 vLLM Triton slot 路径的单层对比最大绝对误差为 `0.00390625`，平均绝对误差约
  `3.53e-4`，通过 BF16 数值容差。
- `moe_layer_index` 错位修复前，graph 输出会在第 12 个 offload 层之后损坏；当前代码
  对 layer 顺序、超额调用和 identity slot 都有回归测试。
- 50 条正式结果只验证固定长度 serving 性能，不验证回答质量或严格端到端 token
  等价。性能和质量结论必须分开。
- 当前 A6000 合同显存余量较小，driver、allocator、模型或 capture size 变化都可能
  OOM；128-slot graph 方案也不是最小显存方案。

## 复现实验

```bash
cd /home/lcw/LatchMoE-GPU
PY=/home/lcw/miniconda3/envs/latchmoe-cuda/bin/python
MANIFEST=benchmark/manifests/offload_manifest.qwen3-base-ad44.first12.graph128.local.json
DATASET=/home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json
```

先校验 graph manifest：

```bash
$PY benchmark/scripts/generate_offload_manifest.py \
  --model /home/lcw/model \
  --revision ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 \
  --layers 0,1,2,3,4,5,6,7,8,9,10,11 \
  --num-slots 128 \
  --output "$MANIFEST" \
  --check
```

运行回归与静态检查：

```bash
LATCHMOE_REAL_MANIFEST="$MANIFEST" LATCHMOE_RUN_REAL=1 $PY -m pytest tests -q
$PY -m ruff check .
$PY -m ruff format --check .
git diff --check
```

正式 benchmark 目录不能已存在。两个 server 必须顺序执行：

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%S)

$PY benchmark/scripts/run_sharegpt.py \
  --mode uva-piecewise \
  --artifact-dir "artifacts/${RUN_ID}-uva-piecewise" \
  --manifest "$MANIFEST" --dataset "$DATASET" \
  --repetitions 3 --num-prompts 50 --output-len 128 \
  --max-concurrency 8 --max-num-seqs 8 \
  --max-model-len 2048 --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 268435456 --port 8026

$PY benchmark/scripts/run_sharegpt.py \
  --mode latchmoe-piecewise \
  --artifact-dir "artifacts/${RUN_ID}-latchmoe-piecewise" \
  --manifest "$MANIFEST" --dataset "$DATASET" \
  --repetitions 3 --num-prompts 50 --output-len 128 \
  --max-concurrency 8 --max-num-seqs 8 \
  --max-model-len 2048 --max-num-batched-tokens 512 \
  --kv-cache-memory-bytes 268435456 --port 8026

$PY benchmark/scripts/compare_sharegpt.py \
  --uva-summary "artifacts/${RUN_ID}-uva-piecewise/summary.json" \
  --latchmoe-summary "artifacts/${RUN_ID}-latchmoe-piecewise/summary.json" \
  --output "artifacts/${RUN_ID}-piecewise-comparison.json"
```

runner 会拒绝少于 3 轮、不是 50 条、合同不一致、请求失败、offload 字节量不一致，或
PIECEWISE profile 缺少 capture/replay 的最终结果。它还会关闭 compile cache，避免旧
AOT artifact 绕过新的 layer adapter，并清除本地请求的 proxy 环境。

## 正式 artifacts

- `artifacts/sharegpt-c8-uva-piecewise-20260715-r1/`
- `artifacts/sharegpt-c8-latchmoe-piecewise-20260715-r1/`
- `artifacts/sharegpt-c8-piecewise-comparison-20260715-r1.json`

## 本地 Qwen3-30B INT8 图重放消融

单张 RTX A6000 不能把原始 BF16 checkpoint（约 57 GB）完整放入显存。本仓库的
`benchmark/scripts/run_int8_graph_ablation.py` 使用 vLLM `experts_int8` 在加载期间量化
Qwen3-MoE 的专家矩阵；其余很小的 dense/router 权重仍以 BF16 计算，且没有设置
`--cpu-offload-gb`。这使完整模型常驻 GPU，显存占用约 29.96 GiB。eager 与 PIECEWISE
两组共享同一模型、请求、调度、KV 和 `VLLM_COMPILE` 配置，唯一运行时变量是
`cudagraph_mode=NONE/PIECEWISE`。这里没有使用 `--enforce-eager`，因为它还会关闭
`torch.compile`，会引入第二个控制变量。

```bash
RUN_ID=$(date -u +%Y%m%dT%H%M%S)
$PY benchmark/scripts/run_int8_graph_ablation.py \
  --artifact-dir "artifacts/${RUN_ID}-int8-eager-graph"
$PY benchmark/scripts/plot_int8_graph_ablation.py \
  "artifacts/${RUN_ID}-int8-eager-graph/results.json" \
  "artifacts/${RUN_ID}-int8-eager-graph/int8_eager_vs_graph.png"
```

2026-07-22 本机运行使用 ShareGPT 50 条、每条生成 128 token、并发 8、3 轮测量。eager
中位 TPOT 为 35.294 ms/token（171.450 token/s），PIECEWISE 为 27.800 ms/token
（217.332 token/s），分别降低 21.23% 和提高 26.76%。两组均完成 50/50 请求、输出
6,400 token；graph server 日志记录了 1/2/4/8/16-token PIECEWISE capture，capture
额外占用约 0.07 GiB。结果目录和图由脚本保存在 `artifacts/<run-id>-int8-eager-graph/`。

上一版绘图把完整 TPOT 近似为 Device Execution，并把 Host-induced Device Gaps 设为
0；它只能用于端到端 graph 对比，不能用于解释 CUDA timeline。需要做时间分解时使用
下面的 profiler 实验。

### CUDA timeline 四数据集实验

`prepare_profiler_datasets.py` 固定 seed 42，为 ShareGPT、LongBench `2wikimqa`、
HumanEval 和 GSM8K 各准备 50 条 custom JSONL。`run_int8_profiler_ablation.py` 对每个
数据集分别运行 `cudagraph_mode=NONE/PIECEWISE`，每个 case 启动独立 server，并在
warmup 后通过 `/start_profile` 采集 PyTorch/CUPTI CUDA timeline。两侧都保留
`VLLM_COMPILE`、INT8、调度、KV cache 和请求合同，唯一运行时变量为 graph replay。

对每个 `execute_context_0(0)_generation_N(N)` 纯 decode iteration，Device Execution
定义为设备实际执行 kernel 的忙碌时间：分析器只对区间内 `kernel` 时间取并集，包括
Attention、GroupedMatmul/MoE、Norm 等设备计算，明确不包含 `gpu_memcpy` 和
`gpu_memset`。跨 CUDA stream 的重叠 kernel 不会重复计时。

Host-induced Device Gaps 定义为平均 TPOT 减去上述 kernel-only Device Execution，表示
设备没有执行 kernel 的剩余关键路径时间；其中包括 host 调度、算子下发、运行时同步、
graph replay 发起、`gpu_memcpy`/`gpu_memset` 以及这些行为造成的设备空闲。两部分只分解
decode TPOT，均不包含 prefill 和 TTFT。

```bash
$PY benchmark/scripts/prepare_profiler_datasets.py \
  --sharegpt /home/lcw/datasets/ShareGPT_V3_unfiltered_cleaned_split.json \
  --model /home/lcw/model --output-dir artifacts/profiler-datasets \
  --num-samples 50 --max-input-tokens 768 --seed 42

RUN_ID=$(date -u +%Y%m%dT%H%M%S)
$PY benchmark/scripts/run_int8_profiler_ablation.py \
  --dataset-dir artifacts/profiler-datasets \
  --artifact-dir "artifacts/${RUN_ID}-int8-cuda-timeline" \
  --num-prompts 50 --output-len 128 --max-concurrency 8 \
  --profile-delay-iterations 16 --profile-iterations 80
$PY benchmark/scripts/plot_profiler_ablation.py \
  "artifacts/${RUN_ID}-int8-cuda-timeline/results.json" \
  "artifacts/${RUN_ID}-int8-cuda-timeline/int8_cuda_timeline_ablation.png"
```

2026-07-22 本机结果位于
`artifacts/20260722T113657-int8-cuda-timeline/`。8 个 case 均完成 50/50 请求，每份
trace 包含 80 个纯 decode iteration。eager 四组的平均 Host-induced Device Gaps 为
28.64、35.09、36.88、40.49 ms；graph 分别为 0.46、0.90、0.42、0.51 ms。graph
trace 每份包含 3,920 次 `cudaGraphLaunch`，eager trace 为 0 次。

每个 case 目录都保留原始 `.pt.trace.json`、`timeline_summary.json`、client/server log
和启动命令；顶层 `contract.json`、`results.json` 与数据集 manifest 绑定了模型、请求和
profiler 合同。

历史 eager artifact 仍保留在：

- `artifacts/sharegpt-c8-uva-20260715/`
- `artifacts/sharegpt-c8-latchmoe-eager-20260715/`
- `artifacts/sharegpt-c8-comparison-20260715.json`

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
