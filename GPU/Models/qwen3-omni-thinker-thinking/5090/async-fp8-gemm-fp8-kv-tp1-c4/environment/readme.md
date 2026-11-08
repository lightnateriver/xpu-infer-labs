# 环境版本记录

实测为 RTX 5090 32 GiB 单卡、Linux x86_64、Python 3.10 原生 venv，vLLM/vLLM-Omni 都为 0.30.0。CUDA Toolkit 13.0 / nvcc 13.0.48、GCC 11.4.0，配合仓库 runtime 兼容层和原 K/V scales。

`versions.json` 与 `reference-versions.txt` 是从实测环境记录提取的主依赖版本；不是完整 pip lock、官方基础镜像承诺或包二进制一致性证明。没有可验证的原包构建 Git commit，仓库也不提供私有镜像地址。

建议激活等价且已验证的环境，再执行：

```bash
python3 scripts/check_environment.py
```

需要完整 CUDA 13 工具链（nvcc、头文件、可用运行库）、GCC 11 和 ffmpeg/ffprobe；只有 CUDA runtime 而缺少 nvcc 会导致 JIT 失败。依赖应从有权限的匹配 wheel 集合/环境准备，不能仅凭同版本号推断完整功能一致。

启动/测试命令、FP8/Graph/async 配置以及历史 3.024 条/分钟结果见 [方案说明](../readme.md)。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
