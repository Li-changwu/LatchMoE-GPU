# LatchMoE-GPU Equal-HBM 正式性能实验报告

日期：2026-09-16

## 1. 结论摘要

在单张 NVIDIA RTX A6000 上，使用相同模型、checkpoint、20 个 offloaded layers、参数集合、prompt workload、采样参数、并发度、KV reserve、eager graph policy、源码状态和有效 backend HBM cache 后，LatchMoE async 相对 exact-selection UVA 的三轮正式结果为：

| 指标（三轮中位数） | exact-UVA | LatchMoE async | 变化 |
| --- | ---: | ---: | ---: |
| Output throughput | 8.989 tok/s | 24.460 tok/s | +172.10%，2.721x |
| Request throughput | 0.0702 req/s | 0.1911 req/s | +172.10%，2.721x |
| Median TTFT | 12,343.62 ms | 2,567.44 ms | -79.20% |
| Mean TTFT | 23,551.24 ms | 5,757.47 ms | -75.55% |
| P99 TTFT | 93,975.95 ms | 22,764.59 ms | -75.78% |
| Median TPOT | 649.95 ms | 278.91 ms | -57.09% |
| Mean TPOT | 682.13 ms | 272.33 ms | -60.08% |
| P99 TPOT | 1,067.29 ms | 453.32 ms | -57.53% |

结果说明 LatchMoE 不只是提高了稳态生成吞吐；首 token、持续 decode 和尾延迟都同时改善。TTFT 改善幅度大于 TPOT，表明该 workload 下 prefill/排队阶段从 Main Cache、显式搬运和复用中得到的收益更大。TPOT 仍降低 57.09%，说明收益没有局限于首轮 prefill。

该结论是完整 LatchMoE async 路径相对 exact-UVA 的系统级结果，不能将 +172.10% 单独归因于统一 token-expert layout。统一 layout 的独立证据仅是组织阶段微基准约 3.4x；正式路径还同时包含 Main Cache hit reuse、显式 H2D、victim replacement 和 H2D/compute overlap。

## 2. 对照变量

两组实验使用 commit `0092ba03dd21c11bf949abf4898756710cba274c`，source state SHA-256 均为 `da620cfa4faabb171ea62e3f27a02cc609ea03bcfcdd2f7c61ee88dc2b3ee0ad`。

| 合同项 | 固定值 |
| --- | --- |
| GPU | NVIDIA RTX A6000，单卡 |
| 模型 | `/root/models/Qwen3-30B-A3B-Instruct-2507` |
| 模型 revision | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| dtype / TP | BF16 / TP=1 |
| selection | `midpoint_stratified_v1` |
| selected layers | `1,3,6,8,10,13,15,18,20,22,25,27,30,32,34,37,39,42,44,46` |
| 实际 offload | 24,159,191,040 bytes（22.5 GiB） |
| backend HBM cache | 6,039,797,760 bytes（5.625 GiB） |
| KV reserve | 268,435,456 bytes（256 MiB） |
| graph policy | eager，两侧均无 graph capture/replay |
| dataset | ShareGPT，SHA-256 `35f0e213ce091ed9b9af2a1f0755e9d39f9ccec34ab281cd4ca60d70f6479ba4` |
| workload | 50 prompts，14,278 input tokens/轮，6,400 output tokens/轮 |
| generation | 128 tokens/request，`ignore_eos=true`，temperature 0，seed 42 |
| serving | request rate `inf`，concurrency 8，`max_num_seqs=8` |
| limits | `max_model_len=2048`，`max_num_batched_tokens=512` |
| cache policy | prefix cache 关闭，compile cache 关闭 |
| repetitions | 每组 2 warmup requests + 3 measurement rounds |

UVA 通过一个生命周期覆盖完整 server run 的 GPU tensor，真实保留 6,039,797,760 bytes HBM；LatchMoE 使用相同字节数作为 20 层、每层 32 slots 的 Main Cache。这里相等的是 `backend_hbm_cache_bytes`。审计报告中的 `reservation_equal=false` 只表示 LatchMoE 没有使用名为 `uva_reservation_bytes` 的占位 tensor，而是把同等 HBM 用于真实 Main Cache，不表示两侧有效 HBM 预算不相等。

唯一有意变化的主变量是 offloaded expert weight 的执行路径：

- exact-UVA：从 pinned host UVA view 直接访问同一组 expert 参数。
- LatchMoE async：按 route 将所需 expert 显式搬入逐层 Main Cache，执行 hit reuse、replacement、多 wave 和经 CUDA event 验证的 H2D/compute overlap。

## 3. 三轮原始结果

### 3.1 exact-UVA

| Round | Duration (s) | Output tok/s | Median TTFT (ms) | Median TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 717.13 | 8.925 | 12,288.21 | 647.73 | 93,975.95 | 1,067.29 |
| 2 | 703.60 | 9.096 | 12,343.62 | 658.52 | 86,144.16 | 1,053.95 |
| 3 | 711.94 | 8.989 | 12,504.18 | 649.95 | 94,281.14 | 1,152.51 |

三轮 output throughput 平均值为 9.003 tok/s，标准差 0.087 tok/s，离散系数约 0.96%。

### 3.2 LatchMoE async

| Round | Duration (s) | Output tok/s | Median TTFT (ms) | Median TPOT (ms) | P99 TTFT (ms) | P99 TPOT (ms) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 261.65 | 24.460 | 3,019.95 | 275.95 | 26,790.92 | 451.34 |
| 2 | 268.68 | 23.820 | 2,567.44 | 292.00 | 22,764.59 | 485.64 |
| 3 | 260.83 | 24.537 | 2,562.02 | 278.91 | 22,105.68 | 453.32 |

三轮 output throughput 平均值为 24.273 tok/s，标准差 0.394 tok/s，离散系数约 1.62%。两组吞吐离散系数均低于 2%，没有单轮异常值主导中位数。LatchMoE 的 TTFT 跨轮波动高于 UVA，主要来自第 1 轮 median TTFT 为 3,019.95 ms，而后两轮约 2,565 ms；吞吐和 TPOT 仍保持稳定。

## 4. 指标分析

### 4.1 吞吐

LatchMoE 的 output throughput 中位数为 24.460 tok/s，是 UVA 8.989 tok/s 的 2.721 倍。每轮输出 token 总数固定为 6,400，因此该差异与总 duration 一致：UVA 三轮 duration 的中位数约 711.94 s，LatchMoE 约 261.65 s。

这是本次实验最稳健的主结论。两组都完成 3/3 轮，每轮 `completed=50`、`failed=0`，且三轮吞吐离散较低。

### 4.2 TTFT：prefill、排队与首 token

Median TTFT 从 12.34 s 降至 2.57 s，降低 79.20%；P99 TTFT 从 93.98 s 降至 22.76 s，降低 75.78%。首 token 延迟的改善显著大于 TPOT，说明在并发 8 和 ShareGPT 长短 prompt 混合 workload 下，UVA 的 host-weight 访问会明显放大 prefill 与排队时间。

LatchMoE 将当前 route 的 expert 权重显式搬入 GPU Main Cache，并在后续 token/layer forward 中复用。profile 证明 overlap 实际发生，但本实验没有分别关闭 cache reuse、replacement 或 overlap，因此只能把 TTFT 改善归因于完整 LatchMoE async 路径，不能从这组数据分解各机制的独立贡献。

### 4.3 TPOT：持续 decode

Median TPOT 从 649.95 ms 降至 278.91 ms，降低 57.09%；P99 TPOT 从 1,067.29 ms 降至 453.32 ms，降低 57.53%。中位数和 P99 改善幅度接近，说明 LatchMoE 不仅改善平均 decode，还压低了慢 token 的尾部延迟。

TPOT 改善小于 TTFT 是合理的：decode 每步 token 数较少，可利用的批量和 prefill 复用模式不同，但 Main Cache 命中与 H2D/compute overlap 仍持续减少 UVA host access 的开销。

### 4.4 Profile 证据

发布前 verifier 对 LatchMoE profile 的检查结果为：

| 证据项 | 结果 |
| --- | ---: |
| 总 profile events | 123,379 |
| Main Cache forwards | 42,450 |
| unified layout forwards | 42,450 / 42,450 |
| overlap candidates | 80,869 |
| actual overlaps | 80,869 / 80,869 |
| temporary bank bytes | 0 |

每个 `main_cache_waves` event 都声明 `pair_layout="unified_token_expert_v1"` 和一次 layout 构建；所有 overlap candidate 都有相交的 CUDA event 时间窗，而不是只启用配置开关。该证据证明被测候选确实执行了预期算法路径。

## 5. 可解释范围与限制

1. 本结果是 A6000 上 exact-UVA 与 LatchMoE async 的受限两方对照。full-resident native 因显存不足保持 `hardware-blocked`，不能宣称 native-equivalent 性能或正确性。
2. 正式性能 workload 之前已在冻结 correctness manifest 上完成 exact-UVA、serial LatchMoE 和 async LatchMoE 的 4/4 逐 token 一致性门槛。这里的 50-request ShareGPT benchmark 用于性能测量，不替代该正确性门槛。
3. `latchmoe-eager` 表示 graph policy 为 eager，不表示关闭异步搬运；本轮 `overlap_enabled=true`，并有 80,869 个真实 overlap 窗口。
4. +172.10% 是完整系统路径收益。要量化统一 layout、cache reuse 和 overlap 的单项贡献，需要在同一 source state 下增加 `overlap=0`、旧 per-wave mask 和不同 slot 数的消融实验。
5. workload 固定为 concurrency 8、50 prompts 和 128 输出 tokens；结果不能直接外推到 concurrency 1、高并发饱和区、超长 prefill、不同 route locality 或其他 GPU。

## 6. Artifact 索引

- [exact-UVA summary](../artifacts/20260915T-formal-task12-equalhbm-uva-r1/summary.json)
- [LatchMoE summary](../artifacts/20260915T-formal-task12-equalhbm-latchmoe-r1/summary.json)
- [严格比较报告](../artifacts/20260915T-formal-task12-equalhbm-comparison.json)
- [预算与 profile 审计](../artifacts/20260915T-formal-task12-equalhbm-budget-verification.json)
- [实施与验收计划](2026-09-10-latchmoe-gpu-main-cache-alignment.md)

两组 run 的 `SHA256SUMS` 均已通过，比较器输出 `comparable_contract=true` 和 `comparable_offload_bytes=true`，budget verifier 输出 `PASS`。

## 7. NPU 参数对齐补充实验

concurrency=1、12 offload layers、3.375 GiB Main Cache、NPU 原始 20 条 prompt 的后续 GPU 单轮实验见 [GPU/NPU 参数对齐报告](2026-09-16-latchmoe-npu-aligned-gpu-benchmark.md)。该实验中 GPU 为 6.792 tok/s，NPU 为 13.298 tok/s；但两侧 checkpoint、输出 token 总数和总 offload bytes 未完全一致，因此结果标记为 exploratory，不能覆盖本报告的 GPU 内部 equal-HBM 正式结论。
