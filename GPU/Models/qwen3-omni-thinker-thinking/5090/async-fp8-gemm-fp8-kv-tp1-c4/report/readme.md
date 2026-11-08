# 独立性能报告

下载并用浏览器打开 [report.html](report.html)。GitHub 的文件页通常展示源码；报告不依赖网络、外部资源或其他 HTML。

内容包括卡型/版本、3.024 条/分钟与 238.54 token/s/卡、TTFT/TPOT/E2E、逐轮及匿名逐样本性能、吞吐柱状图、FP8/Graph/async/KV 配置和输入/计时方法，并保留同配置复测波动与精度边界。

重建方式（只读取随包匿名数据，不运行推理）：

```bash
python3 report/build_report.py
```

报告不包含原业务 Caption、完整 prompt、私有数据来源或机器登录信息。真实输入和启动/测试方法见 [方案说明](../readme.md)。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
