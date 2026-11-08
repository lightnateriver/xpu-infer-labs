# 匿名证据

- `results.json`：7方案/22轮/330条匿名指标、默认选择、显存核算、Graph检查和限制。
- `sample-metrics.csv`：逐条计时、Thinking开关观察、输入/输出tokens和完整性；匿名ID在各轮保持一致。原C4一条ID不匹配保留为无效，其它329条通过协议校验。
- `request-counter-audits.json`：每轮15条请求增量与服务身份连续性。
- `kv-timelines.json`：相对秒数、KV占用和相对抢占计数；1Hz，不是GPU逐毫秒profiling。
- `admission-events.json`：匿名FIFO准入、预算和排队事件；epoch和原始ID不公开。
- `captured-shapes.json`、`graph-forward-checks.json`：捕获26张与遇到形状的逐元素前向检查，分开计数。
- runtime/KV/FP8审计：只保留计算身份、数值和精度字段；无真实机器标识。
- `publication-provenance.json`：关键Worker/客户端源码LF文本SHA身份、新包装脚本未GPU复测的边界。

不包含媒体、prompt、Caption或Thinking正文、私有来源/路径/登录和机器信息。原始私有证据在本地保留。完整性/逐元素检查不是Caption语义精度；所有正式尝试进入主吞吐分母，不只挑选较快轮次。


[返回方案](../readme.md)

静态发布检查摘要见[publication-checks.json](publication-checks.json)，不包含本地路径或访问身份。
