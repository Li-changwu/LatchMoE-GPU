# NPU 对齐配置下 A6000 LatchMoE 与 UVA 对照

日期：2026-09-16

## 1. Correctness 配置

| 配置项 | 值 |
| --- | --- |
| GPU | NVIDIA RTX A6000 48 GB，单卡 |
| 模型 | `/root/models/Qwen3-30B-A3B-Instruct-2507` |
| 模型 revision | `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| 后端 | exact-UVA、serial LatchMoE、async LatchMoE |
| Prompt 数 | 4 |
| Sampling | temperature=0 |
| Max model len | 4096 |
| Max num seqs | 1 |
| KV reserve | 536,870,912 bytes |
| Graph | PIECEWISE，capture size=1 |
| Selected layers | `2,6,10,14,18,22,26,30,34,38,42,46` |
| Manifest host bytes | 14,495,514,624 bytes |
| Residual UVA | 10,928,355,328 bytes |
| Backend HBM cache | 3,623,878,656 bytes |
| Actual offload | 25,423,869,952 bytes |
| Plan ID | `a63cf79af170918e613226861f3b49ffc272db95dece32d954a537b375cebe9c` |
| Source state | `308b55e911e80f4d4c079903d768e1d1d7f3a970a204ecbf93c8248b268b3ba7` |

## 2. Correctness 结果

| 结果项 | 数值 |
| --- | ---: |
| exact-UVA vs serial token IDs | 4/4 |
| exact-UVA vs serial text | 4/4 |
| exact-UVA vs async token IDs | 4/4 |
| exact-UVA vs async text | 4/4 |
| Source state match | true |
| Offload bytes match | true |
| Async actual overlap events | 60 |
| Gate pass | true |
| Qualification | `three_way_4_of_4_exact_token_parity` |

## 3. 性能实验配置

| 配置项 | 值 |
| --- | --- |
| 后端 | exact-selection vLLM UVA、async LatchMoE |
| 启动方式 | 每个后端一次独立完整冷启动 |
| Workload | NPU 证据包原始 20 条 prompt，原顺序 |
| Input tokens | 7,322 |
| Concurrency | 1 |
| Request rate | inf |
| Warmup | 0 |
| Sampling | temperature=0，允许 EOS |
| Max output | 128 tokens |
| Max model len | 4096 |
| Max batched tokens | 4096 |
| Max num seqs | 1 |
| KV reserve | 536,870,912 bytes |
| Graph | PIECEWISE，capture size=1，compile cache disabled |
| Selected layers | `2,6,10,14,18,22,26,30,34,38,42,46` |
| Manifest host bytes | 14,495,514,624 bytes |
| Residual UVA | 10,928,355,328 bytes |
| Actual offload | 25,423,869,952 bytes |
| Backend HBM cache | 3,623,878,656 bytes |
| Workload hash | `c5e8705fd3b838be3e0f9e5e87e6eebf81ec2928219a7a533cfc6bcb4978f83e` |
| Source state | `308b55e911e80f4d4c079903d768e1d1d7f3a970a204ecbf93c8248b268b3ba7` |

## 4. 性能结果

| 指标 | vLLM UVA | LatchMoE GPU | 变化 |
| --- | ---: | ---: | ---: |
| Completed / failed | 20 / 0 | 20 / 0 | - |
| Input tokens | 7,322 | 7,322 | 0 |
| Output tokens | 1,634 | 1,923 | +17.69% |
| Output throughput | 4.569 tok/s | 6.800 tok/s | +48.81% |
| Request throughput | 0.05593 req/s | 0.07072 req/s | +26.45% |
| Mean TTFT | 3,009.91 ms | 2,056.88 ms | -31.66% |
| Median TTFT | 2,211.95 ms | 1,482.22 ms | -32.99% |
| P99 TTFT | 7,981.27 ms | 5,203.82 ms | -34.80% |
| Mean TPOT | 184.23 ms | 128.96 ms | -30.00% |
| Median TPOT | 184.21 ms | 127.21 ms | -30.94% |
| P99 TPOT | 184.42 ms | 144.59 ms | -21.60% |
| Mean E2E | 17,879.42 ms | 14,139.47 ms | -20.92% |
| Median E2E | 24,608.75 ms | 17,672.79 ms | -28.18% |
| P99 E2E | 31,372.50 ms | 21,353.46 ms | -31.94% |
| Measurement duration | 357.59 s | 282.79 s | -20.92% |

## 5. 输出与路径数据

| 数据项 | 结果 |
| --- | ---: |
| Input length 相同 | 20/20 |
| Output length 相同 | 17/20 |
| Generated text 相同 | 6/20 |
| Output length 不同的请求索引 | `4,10,14` |
| Main Cache wave events | 238 |
| Unified layout events | 238/238 |
| Actual overlap events | 252 |
| PIECEWISE capture / replay | 1 / 1 |
| Artifact checksum | 全部通过 |
| 性能结果资格 | `correctness_gate_passed_benchmark_output_not_equivalent` |

## 6. Artifacts

- [三方 exact-token gate](../artifacts/20260916T-npu-aligned-three-way-token-gate.json)
- [UVA summary](../artifacts/20260916T-npu-aligned-gated-gpu-uva-piecewise-r1/summary.json)
- [UVA raw result](../artifacts/20260916T-npu-aligned-gated-gpu-uva-piecewise-r1/repetition-01/raw_result.json)
- [LatchMoE summary](../artifacts/20260916T-npu-aligned-gated-gpu-latchmoe-piecewise-r1/summary.json)
- [LatchMoE raw result](../artifacts/20260916T-npu-aligned-gated-gpu-latchmoe-piecewise-r1/repetition-01/raw_result.json)
- [性能比较](../artifacts/20260916T-npu-aligned-gated-gpu-uva-vs-latchmoe-comparison.json)
- [输出一致性审计](../artifacts/20260916T-npu-aligned-gated-gpu-output-audit.json)
- [冻结 workload](../artifacts/20260916T-npu-aligned-c1-contract/workload_alignment.json)

## 7. Slot 读写顺序修复后 eager 配置

| 配置项 | 值 |
| --- | --- |
| 后端 | exact-selection vLLM UVA、serial LatchMoE |
| Workload | NPU 证据包原始 20 条 prompt，原顺序 |
| Concurrency / max num seqs | 1 / 1 |
| Sampling | temperature=0，允许 EOS |
| Max output / model len | 128 / 4096 tokens |
| KV reserve | 536,870,912 bytes |
| Graph | eager |
| Attention backend | FLASH_ATTN |
| Batch invariant | false |
| Backend HBM cache | 3,623,878,656 bytes |
| Actual offload | 25,423,869,952 bytes |
| Plan ID | `a63cf79af170918e613226861f3b49ffc272db95dece32d954a537b375cebe9c` |
| Source state | `83d5c40e89a628fda7a8c4e8df520fd1f2ecd626548d137224dafca7d333be85` |

## 8. Slot 读写顺序修复后结果

| 指标 | vLLM UVA | serial LatchMoE |
| --- | ---: | ---: |
| Completed / failed | 20 / 0 | 20 / 0 |
| Input tokens | 7,322 | 7,322 |
| Output tokens | 1,753 | 1,753 |
| Output throughput | 4.599 tok/s | 5.920 tok/s |
| Request throughput | 0.05247 req/s | 0.06755 req/s |
| Median TTFT | 2,206.63 ms | 1,499.77 ms |
| Median TPOT | 185.37 ms | 146.93 ms |
| Measurement duration | 381.15 s | 296.10 s |
| Input length match | 20/20 | 20/20 |
| Output length match | 20/20 | 20/20 |
| Generated text match | 20/20 | 20/20 |
| Main Cache wave events | - | 239 |

| Exact-token gate | 结果 |
| --- | ---: |
| exact-UVA vs serial token IDs | 4/4 |
| exact-UVA vs async token IDs | 4/4 |
| Tokens per request | 32 |
| Async Main Cache wave events | 46 |
| Async actual overlap events | 56 |
| Gate | PASS |

- [Token gate audit](../artifacts/20260916T-slot-read-order-token-gate.json)
- [UVA eager summary](../artifacts/20260916T-slot-read-order-npu-aligned-uva-r1/summary.json)
- [Serial LatchMoE eager summary](../artifacts/20260916T-slot-read-order-npu-aligned-latchmoe-r1/summary.json)

## 9. Token 修复后正式对比配置

| 配置项 | 值 |
| --- | --- |
| GPU | NVIDIA RTX A6000 48 GB，单卡 |
| 模型 | `/root/models/Qwen3-30B-A3B-Instruct-2507` |
| 后端 | exact-selection vLLM UVA、async eager LatchMoE |
| 启动方式 | 每个后端一次独立完整冷启动 |
| Workload | NPU 证据包原始 20 条 prompt，原顺序 |
| Concurrency / max num seqs | 1 / 1 |
| Request rate / warmup | inf / 0 |
| Sampling | temperature=0，允许 EOS |
| Max output / model len | 128 / 4096 tokens |
| Max batched tokens | 4096 |
| KV reserve | 536,870,912 bytes |
| Graph policy | eager，compile cache disabled |
| Attention backend | FLASH_ATTN |
| Batch invariant | false |
| Selected layers | `2,6,10,14,18,22,26,30,34,38,42,46` |
| Manifest host bytes | 14,495,514,624 bytes |
| Residual UVA | 10,928,355,328 bytes |
| Actual offload | 25,423,869,952 bytes |
| Backend HBM cache | 3,623,878,656 bytes |
| Plan ID | `a63cf79af170918e613226861f3b49ffc272db95dece32d954a537b375cebe9c` |
| Workload hash | `1b554c18d8d3841391a1a7ff69653d00b8e39e7ce36b5740935dfca4c2830734` |
| Source state | `ee2ce2e152383e407837aa26be7f95871332cc43446f15286665c1caf0937392` |

## 10. Token 修复后正式结果

| 指标 | vLLM UVA | async eager LatchMoE | 变化 |
| --- | ---: | ---: | ---: |
| Completed / failed | 20 / 0 | 20 / 0 | - |
| Input tokens | 7,322 | 7,322 | 0 |
| Output tokens | 1,753 | 1,753 | 0 |
| Output throughput | 4.601 tok/s | 5.912 tok/s | +28.48% |
| Request throughput | 0.05250 req/s | 0.06745 req/s | +28.48% |
| Mean TTFT | 2,994.80 ms | 2,082.80 ms | -30.45% |
| Median TTFT | 2,209.62 ms | 1,502.10 ms | -32.02% |
| P99 TTFT | 7,988.88 ms | 5,260.87 ms | -34.15% |
| Mean TPOT | 185.28 ms | 148.23 ms | -19.99% |
| Median TPOT | 185.28 ms | 147.22 ms | -20.54% |
| P99 TPOT | 185.49 ms | 154.26 ms | -16.84% |
| Mean E2E | 19,048.35 ms | 14,826.39 ms | -22.16% |
| Median E2E | 24,886.43 ms | 19,697.74 ms | -20.85% |
| P99 E2E | 31,516.10 ms | 23,953.30 ms | -24.00% |
| Measurement duration | 380.97 s | 296.53 s | -22.16% |

| 输出与执行数据 | 结果 |
| --- | ---: |
| Input length match | 20/20 |
| Output length match | 20/20 |
| Generated text match | 20/20 |
| Mismatched requests | `[]` |
| Main Cache wave events | 239 |
| Unified layout events | 239/239 |
| Actual overlap events | 260 |
| CUDA graph capture / replay | 0 / 0 |
| Comparable contract | true |
| Comparable offload bytes | true |
| Artifact checksums | PASS |
| Qualification | `npu_aligned_equal_hbm_20_of_20_output_equivalent` |

## 11. Token 修复后 Artifacts

- [UVA summary](../artifacts/20260916T-tokenfix-npu-aligned-uva-eager-final-r1/summary.json)
- [UVA raw result](../artifacts/20260916T-tokenfix-npu-aligned-uva-eager-final-r1/repetition-01/raw_result.json)
- [LatchMoE summary](../artifacts/20260916T-tokenfix-npu-aligned-latchmoe-async-eager-final-r1/summary.json)
- [LatchMoE raw result](../artifacts/20260916T-tokenfix-npu-aligned-latchmoe-async-eager-final-r1/repetition-01/raw_result.json)
- [性能比较](../artifacts/20260916T-tokenfix-npu-aligned-eager-comparison.json)
- [输出一致性审计](../artifacts/20260916T-tokenfix-npu-aligned-eager-output-audit.json)

## 12. PIECEWISE 重跑数据（不纳入正式结论）

| 数据项 | vLLM UVA | async LatchMoE |
| --- | ---: | ---: |
| Completed / failed | 20 / 0 | 20 / 0 |
| Output tokens | 1,634 | 1,929 |
| Output throughput | 4.578 tok/s | 6.802 tok/s |
| Output length match | 17/20 | 17/20 |
| Generated text match | 8/20 | 8/20 |
| Qualification | `output_not_equivalent` | `output_not_equivalent` |
