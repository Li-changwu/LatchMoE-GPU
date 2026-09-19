# LatchMoE GPU/NPU 参数对齐单轮对照

日期：2026-09-16

## 1. 结论

本轮在 RTX A6000 上复用了 NPU Qwen ShareGPT 证据包中的 20 条原始 prompt，并对齐 concurrency、offload layers、Main Cache、输出上限、EOS、KV、batching 和 selected layer IDs。GPU 完成了 20/20 条请求，但结果不再支持“GPU 约为 NPU 两倍”的判断：

| 指标 | GPU A6000 | NPU Ascend 910B2 | GPU / NPU |
| --- | ---: | ---: | ---: |
| Output throughput | 6.792 tok/s | 13.298 tok/s | 0.511x |
| Median TTFT | 1,485.22 ms | 720.40 ms | 2.062x |
| Median TPOT | 126.94 ms | 69.39 ms | 1.829x |
| Median E2E | 17,624.24 ms | 9,481.56 ms | 1.859x |
| Measurement duration | 283.12 s | 176.34 s | 1.606x |
| Output tokens | 1,923 | 2,345 | 0.820x |

在该单轮 workload 下，NPU output throughput 是 GPU 的 1.958 倍；GPU median TTFT、TPOT 分别是 NPU 的 2.062 倍和 1.829 倍。

但这不是严格的硬件等价比较。两侧 output token 总数相差 422（GPU 少 18.0%），说明 checkpoint 或数值执行结果没有保持 token 等价；同时 A6000 为容纳模型额外使用了 10.18 GiB residual UVA。吞吐和总时长只能作为当前系统配置的 exploratory 观测，不能单独归因于 GPU/NPU 硬件。

## 2. 已对齐变量

| 合同项 | GPU 与 NPU 的共同值 |
| --- | --- |
| 请求 | 同一 20 条 prompt、相同顺序 |
| Prompt tokens | 逐条相同，总计 7,322 |
| 并发与提交 | concurrency=1、request batch size=1 |
| 生成 | temperature=0、最多 128 tokens、允许 EOS |
| 上下文 | `max_model_len=4096` |
| Batching | `max_num_batched_tokens=4096` |
| KV reserve | 512 MiB |
| Selected layers | `2,6,10,14,18,22,26,30,34,38,42,46` |
| Managed expert host bytes | 14,495,514,624 bytes（13.5 GiB） |
| Main Cache | 3,623,878,656 bytes（3.375 GiB） |
| Slots | 每层 32 |
| 重复 | 每配置一次完整启动、单轮 measurement |

GPU 使用预格式化 custom JSONL 和 `--skip-chat-template` 提交 prompt。原因是 vLLM 0.19.1 的 ShareGPT loader 会固定过滤超过 1,024 tokens 的 prompt；直接使用它会丢弃 1,193-token 和 2,003-token 两条请求。Custom loader 已直接验证 20/20 条、逐条 token 长度一致且总数为 7,322。

## 3. 未对齐变量

### 3.1 总卸载量

NPU 64 GB 环境只需管理 12 层专家权重：13.5 GiB host store + 3.375 GiB Main Cache。A6000 48 GB 无法在仅卸载这 12 层的条件下完成启动和长 prefill，因此 GPU 额外用 stock UVA 卸载了 10,928,355,328 bytes（10.18 GiB）非 Main-Cache 参数。

GPU 实际总 offload 为 25,423,869,952 bytes（约 23.68 GiB），NPU 为 14,495,514,624 bytes（13.5 GiB）。额外 residual UVA 会让更多层从 host 直接取权重，是 GPU TPOT 和 TTFT 变慢的重要混杂变量。

### 3.2 Checkpoint 与输出

GPU 使用 `/root/models/Qwen3-30B-A3B-Instruct-2507`，revision `0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`；NPU 证据记录的是 `/root/data/shared_models/strict-models/Qwen3-30B-A3B`，证据包没有给出可与 GPU revision 对等验证的 source hash。

两侧只有 13/20 条请求的 output length 同为 128。GPU 总输出 1,923 tokens，NPU 为 2,345，证明本轮没有达到 exact-token 或至少 output-length parity。因此：

- TPOT 可用于观察各自 decode 服务速度，但仍受不同 token 序列影响。
- Output throughput 和 duration 同时受速度与 EOS 位置影响，不能视为严格等工作量结果。
- 本轮不能替代同 checkpoint 的 exact-token correctness gate。

### 3.3 软件与 graph 路径

GPU 使用 vLLM 0.19.1；NPU 使用 vLLM 0.21.0 和 vLLM Ascend OOT compiler。NPU 证据记录了 wave-prefill graph arena（108 graphs、722 replays）；GPU 使用 PIECEWISE graph，但超过 32 active experts 的 prefill 多波路径在 graph boundary 外执行。GPU 日志还显示 A6000 缺少对应的 unquantized MoE tuning config，使用了默认 kernel config。

这些差异意味着本轮比较的是两套完整系统栈，不是单独比较 PCIe GPU 与 Ascend NPU 的算力。

## 4. GPU 路径证据

通过的 GPU run 为 `20260916T-npu-aligned-c1-latchmoe-piecewise-r8`：

| 证据项 | 结果 |
| --- | ---: |
| completed / failed | 20 / 0 |
| input / output tokens | 7,322 / 1,923 |
| PIECEWISE capture / replay | 1 / 1 |
| Main Cache wave events | 238 |
| Unified layout events | 238 / 238 |
| Actual overlap events | 252 |
| 最大单层 pair count | 16,024 |
| 最大 wave count | 5 |
| Profile failure events | 0 |
| Artifact checksum | `sha256sum -c SHA256SUMS` 全部通过 |

首条 742-token prefill 的统一 layout 产生 5,936 个 token-expert pairs；1,193-token prompt 产生 9,544 pairs。2003-token 长 prompt 同样完成，说明修正后的 production plan/native seam 不再为每个 routed pair 展开一份 FP32 专家权重。

## 5. 执行中发现并修复的问题

1. Diagnostic residual-UVA 原先退化到 manifest/functional seam，首条长 prefill 会申请 12.01 GiB 临时权重。现改为保留 production residency plan 和 vLLM native modular seam。
2. PIECEWISE overflow 分支原先没有把 native seam 传入多波执行器，production plan 会 fail closed。现已传递模块锁定的 seam，并增加 graph overflow 回归测试。
3. ShareGPT loader 静默过滤两个长 prompt。benchmark 驱动新增 exploratory custom/preformatted 模式，合同显式记录 loader 和 `skip_chat_template`。
4. 本地 vLLM custom benchmark loader 缺少 `pandas`；已在 `/root/latchmoe-venv` 安装 benchmark 运行依赖。

最终完整测试集通过 `195 passed`；格式化后的相关集成与 benchmark 聚焦测试通过 `21 passed`。`python -m compileall` 和 `git diff --check` 同样通过。

## 6. 失败轨迹与结果资格

r1-r4 依次证明了严格 12 层 + 3.375 GiB Main Cache 在 A6000 上还需要额外容量，并定位了 functional seam 的 FP32 workspace。r5 定位 graph overflow seam 漏传；r6 完成 18 条但因 ShareGPT loader 过滤而作废；r7 因缺少 custom loader 依赖在发请求前退出；r8 是唯一完整 20/20 的候选结果。

本报告只发布 r8 指标。r1-r7 均保留 `failure.json`、server/client 日志和 profile，用于解释选择过程，不参与性能统计。

## 7. Artifact 索引

- [GPU summary](../artifacts/20260916T-npu-aligned-c1-latchmoe-piecewise-r8/summary.json)
- [GPU raw result](../artifacts/20260916T-npu-aligned-c1-latchmoe-piecewise-r8/repetition-01/raw_result.json)
- [GPU contract](../artifacts/20260916T-npu-aligned-c1-latchmoe-piecewise-r8/contract.json)
- [GPU profile](../artifacts/20260916T-npu-aligned-c1-latchmoe-piecewise-r8/profile.jsonl)
- [Workload alignment](../artifacts/20260916T-npu-aligned-c1-contract/workload_alignment.json)
- [Machine-readable comparison](../artifacts/20260916T-npu-aligned-c1-comparison.json)
- [NPU evidence summary](../npu_evidence/RESULTS.zh.md)
- NPU raw result: `npu_evidence/q1_evidence_20260914.zip:runs/qwen_sharegpt_latchmoe_r1/summary.json`

最终资格：`exploratory_not_strictly_comparable`。若要形成硬件归因结论，下一轮必须使用同一 checkpoint hash，并让两侧 output token IDs 全等；同时需要在 64 GB GPU 或更小模型上移除 residual UVA，使总 offload 也一致。

## 8. A6000 内部 UVA 对照

最新重跑在性能实验前先通过 exact-UVA、serial LatchMoE 和 async LatchMoE 三方 `4/4` 逐 token 门槛；三方实际卸载均为 25,423,869,952 bytes、source state 均为 `308b55e911e80f4d4c079903d768e1d1d7f3a970a204ecbf93c8248b268b3ba7`，async 记录 60 个真实 overlap 窗口。

随后使用本报告冻结的 NPU 对齐 workload，在同一 A6000 上重新完成 equal-offload、equal-backend-HBM 的 async LatchMoE 与 exact-selection vLLM UVA 单轮冷启动对照。LatchMoE output throughput 为 `6.800 tok/s`，UVA 为 `4.569 tok/s`；median TPOT 降低 `30.94%`，median TTFT 降低 `32.99%`。两侧 20 条性能请求的生成文本仍仅 6/20 完全相同，因此 correctness 资格为 `three_way_4_of_4_exact_token_parity`，性能资格单独标为 `correctness_gate_passed_benchmark_output_not_equivalent`。完整合同、指标与限制见 [A6000 LatchMoE/UVA 对照报告](2026-09-16-npu-aligned-a6000-latchmoe-vs-uva.md)。
