# 配置

- `deploy.json`：3.024 条/分钟历史方案的原部署 JSON。TP1/C4、异步调度、96 层 FP8 GEMM、FP8 KV 6.25 GiB、上下文 65536、chunked prefill 4096、Graph `[1,2,4]`。
- `kv-scales.json`：原 48 层 K/V 静态 scales。metadata 记录 320 个校准样本、`heldout_used=false`、max-abs/448 和 10% 余量；不含校准数据。不能无验证地迁移到不同权重。
- `model-fingerprint.json`：实测权重及 tokenizer/processor 等 10 个文件的 SHA256；无本地路径或模型文件。

启动脚本复制部署 JSON 到本地状态目录，设置 `KV_SCALES_FILE` 后启动。`devices: "0"` 是 CUDA_VISIBLE_DEVICES 映射后的逻辑卡 0；物理卡由 `start.sh --gpu N` 指定。

配置 JSON 未改写成 BF16 KV。完整启动、版本、测试方式和指标见 [方案说明](../readme.md)。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
