# 流式 Caption 性能客户端

`bench_caption.py` 和 `native_adapter.py` 与实测版本保持文本一致（LF 规范化）；保留请求内容构建、视频 2 fps/原音频、全部参考图、enable_thinking=true、流式解析、usage/token IDs、计时、每条完整结果保存及 GPU/API 审计。

`_common.py` 只去掉未使用 Docker/legacy 分支和容器 key 读取，锁文件移到忽略的 `.local/`。保留原生进程模式下 HTTP 成功计数与 engine terminal counter 对齐的 idle 判定，因为 native vLLM 不产生 OmniTiming；native_adapter 的 docker 命令接口只是旧客户端的本地 receipt 适配，**不是容器存在的证明**。

每次测试保存私有输入哈希、原始 SSE、事件时间戳、Thinking/正文、完整 JSON、逐条指标、输入 token 预检、GPU 每秒遥测及计数。不要将这些未脱敏的 .local 结果直接提交。

输入合同见 [data](../data/readme.md)，统计定义及验收见 [主说明](../readme.md)。结构检查只针对这一套 Caption 输出协议，并非通用任意 JSON，也不是语义正确率。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
