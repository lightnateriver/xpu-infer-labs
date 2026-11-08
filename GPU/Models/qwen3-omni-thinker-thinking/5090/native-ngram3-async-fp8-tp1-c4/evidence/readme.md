# 匿名实测证据

- `results.json`：无 n-gram/N3/N5 三条件各两轮，共 90 次正式请求，全部保留匿名逐条指标；含 Thinking、均为原生 TP1/C4。
- `sample-metrics.csv`：与 results.json 对应的 90 行 CSV，匿名编号在条件间固定，保留输入语义/system 哈希用于配对。
- `request-counter-audits.json`：六个正式轮次 API 生成计数 15/15、服务未重启且无额外请求的脱敏判定。
- `native-runtime-audit.json`：实际 native Qwen3OmniMoeThinker 模型类、GPUModelRunner、async=true、NgramProposerGPU。
- `native-modality-compat.json`：DeepStack metadata 修复前后源码 SHA。
- `mixed-fp8-loaded-audit.json`：96 目标 FP8 线性层、48 BF16 router、实际 kernel/scale 审计。
- `allocated-kv-cache-audit.json` / `attention-routing-audit.json`：实际 48 层 FP8 KV 与 XQA、BF16 Q。
- `publication-provenance.json`：6 个核心 worker/client 文件与实测文本 SHA 一致；公开包装层改动范围与未复测声明。

原始视频、system prompt、完整 Caption/Thinking、SSE、遥测主机身份与内部证据压缩包不公开。保留的 hashes 用于核对冻结输入，不提供原数据内容或来源。结构检查结果可以复核格式是否变化，不能证明语义精度或“加速无损”。

N3 比 control 同时扩大了 graph capture sizes；N5 capture=[1,2,4,6,12,24]、max=24，与 N3 的 [1,2,4,8,16] 不同；N5 时间晚于 N3，非交错实验。原模型 fingerprint 是继承的已有指纹，不是这里重新全量校验 checkpoint 的记录。

[独立报告](../report/readme.md) · [方法与结论](../readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
