# 公开匿名证据

历史实测：RTX 5090 32 GiB × 1，TP1/C4，vLLM/vLLM-Omni 0.30.0，异步 + FP8 QKV/O + FP8 KV；15 输入 × 2 轮，30/30 协议完整；3.023557 条/分钟、238.535153 token/s/卡，包含 Thinking。

- `results.json`：原两轮指标、合并结果与匿名逐条数据；另外保留同配置复测和 BF16 KV 后续实验的汇总背景，避免把最佳历史值当成保证。
- `sample-metrics.csv`：30 个匿名逐条性能记录；匿名编号在两轮一致。
- `loaded-linear-audit.json`：实际 96 层 FP8 kernel、数据类型与 async scheduler。
- `allocated-kv-cache-audit.json` / `attention-routing-audit.json`：实际 48 层 FP8 KV 分配视图和 FlashInfer/XQA 路径。
- `model-integrity.json`：10 个源模型文件 SHA256 未改写的记录。
- `client-provenance.json`：原始客户端哈希以及发布适配范围。

只发布 allowlist 数值与模型结构，不含内部地址、登录名、GPU UUID/PID、私有目录、原 sample 名、原 caption/prompt/media 或未经脱敏的压缩归档。协议完整不是人工语义精度。

启动、测试及计时定义见 [方案说明](../readme.md)；[报告](../report/report.html) 为独立 HTML。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
