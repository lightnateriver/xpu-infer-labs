# Qwen3-VL-Embedding-2B · Ascend 910B1

单卡 910B1、64 GiB。两套配置：

| 方案 | 配置 | 用途与边界 |
|---|---|---|
| 稳定单遍版 | [stable-singlepass.json](configs/stable-singlepass.json) | eager，保持已验证原 NPU 向量；仍存在对旧 GPU 库的阈值差异 |
| **NPU-SPG v1（单遍图加速版）** | [npu-spg-v1.json](configs/npu-spg-v1.json) | Single-Pass + Graph；同输入余弦 ≥0.99 门槛下的最快候选，存在新增阈值判定变化 |

配置标识：`qwen3-vl-emb-2b-npu-spg-v1`。64 路请求、64 解码进程、max_num_seqs=1。
EngineCore 与解码进程 OMP/Torch=1；先构造引擎和预热 CPU 池，再把**客户端 MM executor** 内部设为 Torch=8。
不能只在 shell 中全局设 OMP=8，也不能只交付 JSON：精度补丁、CPU 解码和图 worker 都是方案的一部分。

## 关键改动

1. C++ 整数 RGB 转换对齐 x86 的常用 YUV420 路径，避免直接使用 ARM FFmpeg 转换引入像素差异；其他格式仍按原路径，BT.2020 CL 使用 FFmpeg zscale。
2. 57 处语言非 QK Norm 的残差、方差、gamma 运算与 deepstack add 保持指定 FP32 边界，BF16 输出；不改 QK fused RoPE / SwiGLU。eager 使用 RPC 补丁，图版在构造/编译前安装补丁。
3. 单遍用正帧数 header 提示采样，但仍顺序解码并验证全部帧、核对实际帧数。未知 header 两遍；错误正提示先失败再回退，合计三遍。未选中帧损坏、尺寸变化也拒绝。
4. SPG 使用 PIECEWISE，bucket 为 256/512/1024/2048；图 worker 在启动前安装 FP32 表达式，实际 capture/replay 计数写入结果。测试最长约 1905 token，长于 2048 的性能未据此验收。

单遍历史两组中位耗时 13.982/12.218→8.192/9.568 秒，合并吞吐 1.4752 倍且向量逐字节一致。
后续图候选历史 7.745/8.334 秒，相对前后 eager 对照合并吞吐 +11.53%；当时相对旧 GPU32 控制为 108.42%。
这些是历史批次；本次公开代码重测另列，不能混用不同轮次的分子和分母。

## 输入与输出合同

官方模型 Qwen3-VL-Embedding-2B，BF16 权重；权重文件 SHA256：
`c73fa9caeddeb3ff831d46c085a7a5708343248ca777e90f2d486964464509c1`。
1 FPS、最多 64 帧，按官方规则取偶数帧，单帧重复为 `[0,0]`；短边 320、尺寸对齐 32，bicubic + antialias。
原 FPS 与采样帧索引传入模型。完整视频 LAST pooling，2048 维 float32 L2 归一化；无音频、字幕或窗口平均。
系统指令固定为官方 `Represent the user's input.`。

prefix caching 关闭、MM processor cache 为 0。每个计时轮重新读取并校验 SHA256、解码、预处理、执行模型；常驻模型和图可以复用，向量结果不复用。OS 页缓存可能已热。
计时覆盖读取到结果物化，不含启动、进程预热、整组预热、编译和捕获；不是 HTTP E2E、纯芯片算力或单请求延迟。

## 文件与交接

| 路径 | 内容 |
|---|---|
| `runtime/qwen_embedding/` | 实际可运行的推理、解码、校验代码 |
| `configs/` | 固定可选配置 |
| `scripts/benchmark.py` | 普通文件或 TAR 清单、进程 barrier、真实视频预热、计时与私有精度对照 |
| `tests/verify_cpu.py` | 不加载模型的输入防错测试 |
| [environment/README.md](environment/README.md) | 测试环境、安装边界 |
| [INPUTS.md](INPUTS.md) | 清单与私有参考格式 |
| [reports/README.md](reports/README.md) | 历史与本次重测的证据入口 |

运行产生的向量、逐样本合同、视频路径和编译源码均留在本地输出目录，不应提交到公开仓库。
测试集含两组互不重叠的各 64 条近阈值视频，已多次参与本项目评估，第二组不等于新盲测。
不公开私有素材和样本级证据；需使用自有输入复核泛化效果。

## 运行

在 [环境说明](environment/README.md) 的已匹配环境中执行：

```bash
python runtime/qwen_embedding/build_rgb_core.py
python tests/verify_cpu.py
export ASCEND_RT_VISIBLE_DEVICES=0 VLLM_PLUGINS=ascend,ascend_model
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
python scripts/benchmark.py --device npu:0 --model /models/Qwen3-VL-Embedding-2B   --manifest /data/frozen-manifest.json --config configs/npu-spg-v1.json --output /results/spg
```

稳定版将配置换为 `stable-singlepass.json`，使用新的输出目录。脚本设置可信本机 callable RPC 所需的序列化选项，不提供网络服务。
编译临时源码目录位于输出目录下，不修改 vLLM 安装源码。CPU RGB `.so` 必须在目标架构编译；不分发 ARM 二进制。

## 精度验收

历史 SPG 最低同输入 GPU/NPU 余弦 **0.999008029**，128 条三轮全部 ≥0.99。
但冻结旧 GPU gallery 的 16,256 个有向非自身比较中，稳定版跨线 12 个、SPG 15 个；SPG 相对原 NPU 改变 7 个判定。
最大新增单维坐标差约 0.0071024，**不是余弦下降 0.0071024**。没有语义标签，跨线不能称为误判率。
若验收要求旧库 0.99 阈值零跨线，两套 NPU 都尚未通过；需要业务校准或进一步数值对齐。
候选三轮重复一致不等于候选与原 NPU 一致。

更多尝试、已撤回的绑核收益和失败轮说明见 [优化记录](reports/optimization-history.md)。

## 本次公开代码重测

SPG 合并吞吐 **8.110 条/秒**，为本轮 GPU32 的 **114.79%**；最低同输入余弦 **0.999008029**，但旧库跨线为 **15/15/15**。完整结果与跨架构像素边界见 [重测报告](reports/retest-20261108.md)。
