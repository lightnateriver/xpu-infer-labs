# Qwen3-VL-Embedding-2B · RTX 5090 CPU32

单卡 5090，保留原 vLLM GPU 算子和 BF16，只把 CPU 解码进程与准备线程从 8 提到 32，改善并发流水线供给。
历史同一组 128 条视频：三轮中位耗时 **31.247751→10.182663 秒，吞吐 3.0687 倍**。
这是特定历史输入与环境下的全管线收益，不保证任意数据也提升三倍；调整请求到达/动态批次也可能引入浮点差异，不称为数值无损。

`configs/gpu32.json` 是优化配置，`gpu8.json` 为配对对照。64 路输入、max_num_seqs=64、CPU 每进程 1 线程，仍用原两遍解码。
本次没有给 GPU 加单遍解码；GPU/NPU 比例表示两套部署流水线吞吐。

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

在本模型目录执行；GPU 选择发生在启动前。

```bash
bash environment/setup.sh
python tests/verify_cpu.py
export CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export TMPDIR=/dev/shm/qwen-run  # 按本机容量选择可写目录
mkdir -p "$TMPDIR"
python scripts/benchmark.py --device cuda:0 --model /models/Qwen3-VL-Embedding-2B   --manifest /data/frozen-manifest.json --config configs/gpu32.json --output /results/gpu32
```

用同一清单将配置改为 `gpu8.json` 并使用新输出目录，完成配对复测。每个输出目录须不存在，失败记录不会被覆盖。
不要在两个配置之间更换模型、输入、计时边界或预热协议。

## 本次公开代码重测

CPU32 合并吞吐 **7.065 条/秒**，是本轮 CPU8 的 **2.553 倍**。数值有小幅批次波动，旧库跨线为 0/1/1。完整结果与跨架构像素边界见 [重测报告](reports/retest-20261108.md)。
