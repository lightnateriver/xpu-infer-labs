# 原实测 runtime workers

卡型：RTX 5090，SM120；vLLM / vLLM-Omni 0.30.0。启动由 [scripts/start.py](../scripts/start.py) 注入本目录 PYTHONPATH，无需安装成全局模块。

- `caption_worker.py`：保留实测中的图像+视频+原音轨 DeepStack 位置掩码修复，并移除该权重压缩配置中过宽的 ignore 规则，断言 48 层专家实际使用 NVFP4。
- `kv_audit_worker.py`：核对物理 GPU、48 层 KV 的实际 dtype/分配视图，加载原 K/V scales，记录实际 FlashInfer/XQA 路由；独立模块包含原校准支持，默认启动不启用校准。
- `mixed_fp8_worker.py`：断言 48 QKV + 48 O 为 FP8、实际 CutlassFP8ScaledMMLinearKernel；router/head 保持 BF16；记录实际 OmniARAsyncScheduler。

三份核心源码按原内容保留。它们针对上述精确包版本和 checkpoint 结构；升级后若断言不匹配应先定位实现差异。原权重不会写回磁盘。

这些文件是兼容与审计实现，不是质量验收证明。性能测试、启动命令与已测指标见 [方案说明](../readme.md)。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
