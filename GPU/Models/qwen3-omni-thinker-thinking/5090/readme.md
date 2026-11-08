# RTX5090 · Qwen3-Omni Thinker / Thinking

## 当前采用方案

[原生N3/K3＋多FULL Graph＋FP8 KV6.5GiB](native-ngram3-multifull-kv6p5-tp1-c4/readme.md)，TP1/C4、chunk4096。三轮45/45完整、零抢占，**3.575有效条/分钟**，完整轮输出TPS/卡中位数**310.22**。同轮C4/KV6.25GiB基线3.240条/分钟，共10次抢占；有效吞吐提升10.33%。全部Thinking计入。

新报告完整呈现C3、90%/95%准入、6.5/6.75GiB容量、chunk2048七组对照、330条匿名指标和KV时间线。KV6.5GiB峰值仍99.10%，不保证所有输入零抢占；完整性不是语义精度。

## 历史方案

| 方案 | 运行时 | 记录值条/min | 记录输出TPS/卡 | 协议说明 |
|---|---|---:|---:|---|
| [Omni Async FP8](async-fp8-gemm-fp8-kv-tp1-c4/readme.md) | vLLM-Omni | 3.024 | 238.54 | 两轮；同配置另测2.815 |
| [原生N3/K3](native-ngram3-async-fp8-tp1-c4/readme.md) | native vLLM | 3.161 | 273.71 | 两轮；输出上限16384 |
| [多FULL＋KV6.5GiB](native-ngram3-multifull-kv6p5-tp1-c4/readme.md) | native vLLM | 3.575 | 310.22 | 至少三完整轮；输出上限24576；TPS为轮中位数 |

各行计量方式和输出协议不同，历史值仅作索引，不能直接归因Graph、KV或n-gram单项提升。最新报告的收益基线为本轮保留多FULL的C4方案，而非旧N3/Omni。

均使用非官方Thinking-heretic-NVFP4衍生权重、96个attention线性层FP8、48层FP8 KV；不是官方BF16 Thinking全模型。每个目录保持configs/runtime/benchmark/scripts/environment/data/evidence/report布局，自备有权限的模型与冻结数据。原环境存在已有修改，详见各方案版本/源码指纹与发布脚本未GPU复测的边界。
