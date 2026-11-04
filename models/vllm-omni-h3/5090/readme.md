# MiniMax-H3 · vLLM-Omni v0.32.0 · RTX 5090

推荐接手配置为 **V1 SP8「精度 OK」**。DiT 主干使用 FP8 W8A8 Linear、BF16 Dense Attention 和双向 BF16 All-to-All，50 个主层在去噪期间常驻 GPU。**SP4「精度备选」**适合单独评估四卡部署，使用 FlashInfer NVFP4 Dense Attention 和双向 FP8 通信。质量标签来自历史样例的人工观感，新增输入仍需复核。

## 配置

| 模块 | V1 SP8「精度 OK」 | SP4「精度备选」 | 原生 BF16 对照 |
|---|---|---|---|
| 5090 / DiT 并行 | 8 卡 · Ulysses SP8 / TP1 / Ring1 | 4 卡 · Ulysses SP4 / TP1 / Ring1 | 8 卡 · TP8 |
| DiT 主干 Linear | FP8 W8A8，在线 per-tensor，Cutlass | FP8 W8A8，在线 per-tensor，Cutlass | BF16 |
| DiT 主干 Attention | Dense BF16 cuDNN | FlashInfer SM120 Dense NVFP4，QK 和 PV | Dense BF16 cuDNN |
| SP 通信 | BF16 双向 All-to-All | FP8 E4M3 双向 All-to-All，`4+4+4+2` 切头，双缓冲与独立 CUDA 通信流 | TP8 通信路径 |
| AdaLN | Rank-owner，精确投影缓存开启 | Rank-owner，精确投影缓存关闭 | 原生精确投影缓存 |
| DiT 显存与执行 | DLO50，50 个主层常驻，0 个主层流式加载，regional compile | DLO50，50 个主层常驻，0 个主层流式加载，regional compile | Eager |
| 文本编码器 / 视频 VAE | FP8 TP8 / 8 卡切片 | FP8 TP4 / 4 卡切片 | BF16 TP8 / 8 卡切片 |
| 稀疏 Attention / 去噪缓存 | 关闭 / 关闭 | 关闭 / 关闭 | 关闭 / 关闭 |

V1 的 `bf16` 模式保留原生 Attention 和 Ulysses 通信路径。补丁同时包含实验分支，发布的启动配置固定选择表中路径。SP4 的 CLI 默认 backend 仍显示 `CUDNN_ATTN`，主干块由 overlay 切到 FlashInfer NVFP4。Refiner 等其余模块沿用原有路径。

## 性能

**8× RTX 5090，1344×768、24 fps，5 秒请求生成 124 帧。** 以下为一次请求发出至完整 MP4 收完的秒数，服务启动和离线解码校验单独计时。I2VA 两套框架按 A 50、A 20、B 50、B 20 的顺序各跑一次，首个请求可能包含首次执行开销。

| I2VA 5 秒 | V1 SP8 50 步 | V1 SP8 20 步 | 原生 BF16 50 步 | 原生 BF16 20 步 |
|---|---:|---:|---:|---:|
| A | 157.262 | 62.575 | 450.666 | 180.394 |
| B | 146.018 | 62.926 | 443.978 | 180.513 |

- I2VA B 15 秒、20 步，V1 SP8 **323.036 s**，首个请求，362 帧，实际视频约 15.083 秒，峰值预留显存 **25,966 MiB**。
- T2VA A 5 秒、50 步，V1 SP8 **139.741 s**，1 次预热后 2 次正式均值。SP4 **112.997 s**，1 次预热后 1 次正式实测，峰值预留 **29,950 MiB**。
- T2VA A 5 秒、20 步，V1 SP8 **59.358 s**，1 次预热后 2 次正式均值。

I2VA 的 V1 与原生 BF16 同时更换了并行、精度和执行路径。SP4 的公开记录覆盖 T2VA，I2VA 性能与画质需单独测量。完整逐样本数值见 [离线性能报告](report/index.html) 和 [脱敏测量数据](report/measurements.json)。

## 环境与启动

[实测版本清单](environment/observed.json) 和 [环境重建说明](environment/setup.md) 随配方提供。实测环境包括 Ubuntu 22.04.5、Python 3.12.16、RTX 5090 32 GB、驱动 580.126.09、PyTorch 2.13.0+cu130、cuDNN 9.20、NCCL 2.29.7、FlashInfer 0.7.2.post1，以及 SP4 JIT 使用的 nvcc 13.4.92。vLLM 和 vLLM-Omni 均为 **0.32.0**。vLLM-Omni tag 解引用后的 commit 为 `5917730a4b02aedb44d374173cdc1b10484ccd52`。

准备带有对应版本的 Linux CUDA 环境和可合法使用的 MiniMax-H3 checkpoint。建议先核对 PyTorch CUDA、GPU SM120、cuDNN/NCCL 和 FlashInfer 导出符号。两套 overlay 都会导入 FlashInfer 的 H3 NVFP4 符号，V1 的 BF16 模式运行时走 BF16 Attention。SP4 首次执行需可用 CUDA Toolkit 和可写 JIT 缓存目录。

```bash
# 在本目录，激活已安装实测依赖的 Python 3.12 环境
git clone --branch v0.32.0 --depth 1 https://github.com/vllm-project/vllm-omni.git ./overlays/upstream
git -C ./overlays/upstream rev-parse HEAD
export OMNI_SOURCE="$PWD/overlays/upstream"
python scripts/prepare_overlay.py v1-sp8 ./overlays/v1-sp8 --source "$OMNI_SOURCE"

# 将占位路径替换成自己的已准备权重目录，选择空闲 GPU
export MODEL_DIR=/path/to/MiniMax-H3
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export H3_OVERLAY="$PWD/overlays/v1-sp8"
bash scripts/serve.sh v1-sp8
```

`prepare_overlay.py` 从干净源码复制包，检查 patch，再校验与实测一致的 5 个文件 SHA-256。它会保留上游文档并复用已安装发行包的版本元数据。补丁通过独立 `PYTHONPATH` 导入，安装环境不需要就地打补丁。原生基线用同一干净源码，`MODEL_DIR` 指向本地 FL2VA checkpoint 入口，执行 `bash scripts/serve.sh native-bf16-tp8`。切换 SP4 时单独准备 `sp4-alternate` overlay，选择四张卡，设置 `CUDA_HOME`，并执行 `bash scripts/serve.sh sp4-alternate`。服务默认绑定本机回环地址。

```bash
# 使用你有权使用的参考图与提示词
python scripts/benchmark.py --image ./reference.jpg --prompt-file ./prompt.txt \
  --steps 20 --duration 5 --seed 42 --output ./runs/i2va-5s.mp4
python scripts/benchmark.py --image ./reference.jpg --prompt-file ./prompt.txt \
  --steps 20 --duration 15 --seed 42 --output ./runs/i2va-15s.mp4
```

脚本会检查视频和音轨、完整解码 MP4，并在本地写同名 JSON。省略 `--image` 提交 T2VA。实际帧数按 H3 的 `17n+5` 边界对齐。公开包提供历史数值、补丁和复测入口，参考图、提示词与模型权重由复测者提供。

## 接手建议

1. **先用 V1 SP8 跑 5 秒、20 步。** 在日志检查 `H3_FP8_LINEAR` 记录 200 个主干 Linear、`H3_V2_MODE bf16`、50 个 DiT 主层常驻和 VAE 8 卡切片。确保服务能返回完整视频和音轨。
2. **在同一份输入上复核质量。** 固定权重、首帧、提示词、seed、画布、步数和 VAE 切片，先和原生 BF16 逐条看人物身份、口型、说话人切换和动作。确认短片后再覆盖 15 秒。对比 20/50 步时保持其余开关一致。
3. **单独复测长片和四卡容量。** SP4 的 50 步 T2VA 已用近 30 GiB 预留显存。部署前按目标输入验证真实峰值、冷启动与多轮延迟。OOM 时优先记录发生阶段，再把调整常驻层数作为独立配置重新测量。
4. **继续优化时做同版本单项对照。** 先采 DiT、VAE、编码阶段时间，再用 GPU trace 看 BF16 All-to-All、Linear 和 Attention。FP8/NVFP4 切换应同时检查时延与画质。
