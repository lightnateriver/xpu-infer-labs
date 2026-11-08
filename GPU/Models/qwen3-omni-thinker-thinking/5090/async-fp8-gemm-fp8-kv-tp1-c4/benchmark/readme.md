# 原流式 Caption 客户端

`bench_caption.py` 和 `native_adapter.py` 按原实测内容保留。请求包含全部参考图、原视频及同步音轨、2 fps、冻结的完整 system prompt；Thinking=true、temperature=0、seed=0、max_tokens=16384。测试入口见 [scripts](../scripts/readme.md)。

`_common.py` 保留 GPU 归属、空闲与 API 计数检查，整理时仅移除了未使用的 Docker/legacy 后端及容器 key 回退，将锁文件移到 `.local/`。`native_adapter.py` 兼容原客户端的 inspect/top/log hooks，实际检查的是本地进程 session，**不代表使用 Docker**。

记录每条完整原文、Thinking、SSE、token IDs、时间戳、输入 SHA256/token 数、JSON 结构、GPU 进程归属和 API 请求计数。原始输出留在用户本地运行目录，公开证据仅含匿名数值。

历史条件：RTX 5090 32 GiB × 1，vLLM/vLLM-Omni 0.30.0，C4，两轮各预热 1 + 正式 15；完整吞吐 3.023557 条/分钟。TTFT/TPOT/E2E 均含 Thinking；完整计时定义与优化配置见 [方案说明](../readme.md)。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
