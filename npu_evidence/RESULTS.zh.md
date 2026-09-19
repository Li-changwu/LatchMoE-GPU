# Q1 重跑结果

已通过基础证据检查：18/18 组；输出对比完成：12/12 项。
完整结果可用：是。

各数据集使用原固定 20 条 Prompt，输出上限 128、允许 EOS；主矩阵每配置选用一组完整运行。GLM HumanEval Native 的预定完整复测另行披露，见 STABILITY.zh.md；不构造跨启动置信区间。
下表 TPOT 为逐请求 TPOT 的中位数；完整延迟为逐请求调用实测墙钟时间的中位数，包含 Prefill 和 Decode。

| 模型 | 数据集 | 方法 | 服务器 | TTFT中位数(ms) | TPOT中位数(ms) | 完整延迟中位数(s) | 输出token总数 | 峰值allocated(GiB) |
|---|---|---|---|---:|---:|---:|---:|---:|
| qwen | sharegpt | latchmoe | vllm-hust-lcw-23rc | 720.40 | 69.39 | 9.482 | 2345 | 47.739 |
| qwen | sharegpt | eager | vllm-hust-lcw-23rc | 914.72 | 230.97 | 30.033 | 2345 | 47.731 |
| qwen | sharegpt | native | vllm-hust-lcw-23rc | 554.50 | 553.13 | 70.802 | 2345 | 47.489 |
| qwen | humaneval | latchmoe | lcw-91 | 1271.71 | 106.23 | 14.913 | 2560 | 47.635 |
| qwen | humaneval | eager | lcw-91 | 1454.15 | 266.04 | 35.299 | 2560 | 47.642 |
| qwen | humaneval | native | lcw-91 | 598.63 | 553.27 | 70.867 | 2560 | 47.384 |
| qwen | gsm8k | latchmoe | lcw-91 | 1185.04 | 109.47 | 15.201 | 2560 | 47.629 |
| qwen | gsm8k | eager | lcw-91 | 1383.92 | 267.37 | 35.347 | 2560 | 47.631 |
| qwen | gsm8k | native | lcw-91 | 587.42 | 553.19 | 70.843 | 2560 | 47.378 |
| glm | sharegpt | latchmoe | vllm-hust-lcw-23rc | 906.22 | 143.04 | 19.056 | 2333 | 44.896 |
| glm | sharegpt | eager | vllm-hust-lcw-23rc | 1022.58 | 269.36 | 35.204 | 2333 | 44.911 |
| glm | sharegpt | native | vllm-hust-lcw-23rc | 718.42 | 557.30 | 71.499 | 2333 | 44.716 |
| glm | humaneval | latchmoe | lcw-91 | 1014.66 | 138.94 | 18.718 | 2560 | 44.735 |
| glm | humaneval | eager | lcw-91 | 1342.82 | 293.87 | 38.782 | 2560 | 44.738 |
| glm | humaneval | native | lcw-91 | 727.73 | 557.60 | 71.543 | 2560 | 44.555 |
| glm | gsm8k | latchmoe | lcw-91 | 1174.03 | 155.41 | 20.802 | 2524 | 44.714 |
| glm | gsm8k | eager | lcw-91 | 1345.86 | 295.73 | 38.845 | 2524 | 44.715 |
| glm | gsm8k | native | lcw-91 | 723.57 | 557.52 | 71.529 | 2524 | 44.534 |

各项速度比 = 对照方法的中位数 / LatchMoE 的中位数，大于 1 表示 LatchMoE 更快。仅输出 token 完全一致的成对结果列入下表。

| 模型 | 数据集 | 对照 | TTFT速度比 | TPOT速度比 | 完整延迟速度比 |
|---|---|---|---:|---:|---:|
| qwen | sharegpt | eager | 1.270× | 3.329× | 3.168× |
| qwen | sharegpt | native | 0.770× | 7.972× | 7.467× |
| qwen | humaneval | eager | 1.143× | 2.504× | 2.367× |
| qwen | humaneval | native | 0.471× | 5.208× | 4.752× |
| qwen | gsm8k | eager | 1.168× | 2.442× | 2.325× |
| qwen | gsm8k | native | 0.496× | 5.053× | 4.660× |
| glm | sharegpt | eager | 1.128× | 1.883× | 1.847× |
| glm | sharegpt | native | 0.793× | 3.896× | 3.752× |
| glm | humaneval | eager | 1.323× | 2.115× | 2.072× |
| glm | humaneval | native | 0.717× | 4.013× | 3.822× |
| glm | gsm8k | eager | 1.146× | 1.903× | 1.867× |
| glm | gsm8k | native | 0.616× | 3.587× | 3.439× |

ShareGPT 保留原服务器的完整结果；HumanEval 与 GSM8K 在容器迁移后的 lcw-91 上完整补跑。两台服务器均为 910B2，但不把本矩阵描述为同一台物理服务器的结果。每个模型/数据集内部的三方法固定同服务器、同卡、同 NUMA：Qwen 为 6 号卡/NUMA4；GLM 原服务器为 5 号卡，新服务器为 7 号卡，均绑定 NUMA2。只计算同一对照组内的速度比。其他卡及并行模型的资源占用见设备快照，不视为整机独占。
Cache-Offload Eager 与 LatchMoE 使用相同缓存策略、容量和异步加载，缓存历史可能不同。Native 是不同搬运组织的系统参考；GLM Native 预取池比 LatchMoE 动态专家槽多 72 MiB。
上述速度比反映 Prefill 与 Decode 两阶段的整体执行差异：LatchMoE 使用新增 Prefill 批次图和既有 Decode 图，Eager 全程 eager。新增 Prefill 批次图的独立增量收益，需要保持 Decode 路径一致的另项消融；本 Q1 不承担该归因。
启动、编译和新增图捕获不计入请求耗时；吞吐分母为整批生成阶段墙钟时间，包含请求间记录进度的开销。

缓存与搬运审计（同策略不表示逐步缓存历史相同）：

| 模型 | 数据集 | LatchMoE misses | Eager misses | LatchMoE H2D(GiB) | Eager H2D(GiB) | 历史摘要相同 |
|---|---|---:|---:|---:|---:|---|
| qwen | sharegpt | 57267 | 57259 | 503.323 | 503.253 | 否 |
| qwen | humaneval | 83541 | 83566 | 734.247 | 734.467 | 否 |
| qwen | gsm8k | 83029 | 83039 | 729.747 | 729.835 | 否 |
| glm | sharegpt | 77455 | 77459 | 1361.514 | 1361.584 | 否 |
| glm | humaneval | 97002 | 97010 | 1705.113 | 1705.254 | 否 |
| glm | gsm8k | 87906 | 87911 | 1545.223 | 1545.311 | 否 |
