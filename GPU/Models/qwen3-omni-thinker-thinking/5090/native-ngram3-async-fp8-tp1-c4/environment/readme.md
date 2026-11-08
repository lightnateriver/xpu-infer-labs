# 已记录环境与一致性边界

Linux x86_64 / Python 3.10 / RTX 5090 32 GiB SM120；CUDA Toolkit 13.0（nvcc 13.0.48）、GCC 11.4.0，ffmpeg/ffprobe。依赖版本见 versions.json 与 reference-versions.txt。

vLLM-Omni 虽安装，但原生 N3 进程设置 VLLM_PLUGINS='' 禁止其模型注册插件，并固定 VLLM_USE_V2_MODEL_RUNNER=0。原生 API 仅文本输出；不宣称支持此配置下的 Talker 音频生成。

`package-source-manifest.json` 保存 8 个关键实际安装 vLLM 源文件的 SHA256。N3 与 N5 的这些文件一致。版本号本身不足以证明源码/二进制环境相同；原环境含此前继承的编译兼容修改，未记录完整包构建 commit，也不提供私有镜像或完整 wheel 锁文件。不能承诺干净 pip 同版本安装即完成复现。

check_environment.py 是必要版本/工具检查，启动后的 native/modality/FP8/KV 审计同样必需。CUDA 13.0 工具链需要完整 include/CCCL 和 nvcc；不以新版 nvcc 替换后继续声称环境对齐。

[运行步骤](../readme.md) · [源码审计](../evidence/readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
