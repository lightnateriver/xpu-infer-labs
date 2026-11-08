# RTX 5090 · 原生 vLLM N3/K3 + Async + FP8

**选用配置：单卡 TP1 / C4，GPU n-gram 匹配长度 3、每轮最多草拟 3 token，Thinking 开启。**

2026-10-21 两轮合并：**3.161 条/分钟、273.71 输出 token/s/卡，TPOT P50 11.48 ms**。这是推荐保留的实验配置，尚未证明稳定优于无 n-gram；N5/K5 的条/分钟仅比 N3 高 0.85%，TPOT 慢 7.27%，因此未保留。

## 实测结果

| 配置 | 完整正式请求 | 条/分钟 | token/s/卡 | TTFT P50/s | TPOT P50/ms | 样本 E2E P50/s |
|---|---:|---:|---:|---:|---:|---:|
| 原生无 n-gram | 30/30 | 3.090268 | 247.214555 | 6.171 | 13.258 | 66.570 |
| N3/K3 · 本方案 | 30/30 | 3.160517 | 273.713046 | 6.428 | 11.485 | 69.337 |
| N5/K5 · 未保留 | 30/30 | 3.187521 | 277.170914 | 6.498 | 12.320 | 70.497 |

- 15 个不同冻结视频 × 2 轮，每轮完整预热 1 条，再 C4 正式测试 15 条。各条件 30/30 无错误、无截断且观察到 Thinking。
- N3 比原生无 n-gram：任务吞吐 **+2.27%**，token/s **+10.72%**，TPOT **−13.38%**；输出 token 数同时增加 **8.26%**。两轮方差及生成长度变化使这些比率不能作为稳定生产收益承诺。
- 无 n-gram capture sizes=[1,2,4]，N3=[1,2,4,8,16]；这是组合配置比较，不是仅改一个标志的严格消融。N5 capture=[1,2,4,6,12,24]、max=24，也改变了图捕获容量；N5 是同日稍后独立测量，未做交错复测。
- **Thinking 包含在 TTFT、TPOT、请求 E2E、样本 E2E、正式整批 wall 和输出 TPS 中**；启动/编译/预热排除在正式吞吐之外。
- **未证明 Caption 语义精度无损。** 结构完整率与镜头格式检查不是人工正确率；详见报告。

## 目录

| 目录 | 内容 |
|---|---|
| [configs](configs/readme.md) | N3、无 n-gram、N5 三套真实参数的可移植版本；冻结 KV scales 与模型指纹 |
| [runtime](runtime/readme.md) | 实测原生 Worker、FP8/KV 审计、多模态 metadata 兼容修复 |
| [scripts](scripts/readme.md) | 参数化启动/停止、环境检查、预热/正式测试、汇总 |
| [benchmark](benchmark/readme.md) | 原始流式请求和计时客户端，保存全部结果的私有本地证据 |
| [environment](environment/readme.md) | 实测依赖版本与 8 个关键已安装源码 SHA256 |
| [data](data/readme.md) | 冻结输入合同与脱敏 manifest 示例 |
| [evidence](evidence/readme.md) | 三种条件 90 次匿名指标、实际 runtime/kernel/KV 审计与计数 |
| [report](report/readme.md) | 独立浅色红系 HTML 报告和生成脚本 |

## 模型与精度范围

权重是 **Qwen3-Omni-30B-A3B-Thinking-heretic-NVFP4**，非官方衍生 checkpoint，不是官方 BF16 Thinking。复用已有方案相同模型 SHA256 指纹；N3/N5 试验未重新全量计算所有权重 SHA。启动包装脚本会再次逐文件检查指纹。

48 个 attention QKV + 48 个 O 层改 FP8 E4M3 W8A8，静态 tensorwise 权重、动态 activation，BF16 输出；实际 `Fp8PerTensorOnlineLinearMethod` / `CutlassFP8ScaledMMLinearKernel`。48 层 MoE 专家保留 NVFP4，router/lm_head/多模态编码器保持原精度。48 层 KV 使用 FP8 E4M3，6.25 GiB，原冻结 K/V scales（未使用 heldout 数据）。

`ngram_gpu` 没有小模型/额外 drafter 权重，不存在“ngram 的 FP8 精度”参数。这里的 FP8 作用于目标模型的上述线性层和 KV；草拟来自已存在 token 序列。

## 实际运行栈

Linux x86_64、RTX 5090 32 GiB（SM120）×1、Python 3.10、CUDA Toolkit 13.0 / nvcc 13.0.48、GCC 11.4.0。vLLM 0.30.0、PyTorch 2.13.0（CUDA 13.0）、Transformers 5.14.1、FlashInfer 0.6.18.post1、compressed-tensors 0.17.0、Triton 3.7.1；需要 ffmpeg/ffprobe。vLLM-Omni 0.30.0 虽安装，**本进程禁用其插件**，走原生 vLLM Thinker / GPUModelRunner / Async 调度路径，仅生成文本。

原环境包含已有编译/兼容修改，不宣称 `pip install` 同版本即可字节一致复现。仓库不提供原私有镜像。先匹配依赖、工具链及关键源码指纹，再启动审计。版本与指纹说明见 [environment](environment/readme.md)。

## 实践：启动 N3

在 Linux GPU 主机激活已匹配的 Python 环境，进入本目录：

```bash
export MODEL_DIR=/path/to/model
export DATA_DIR=/path/to/private/dataset
export PREPARED_DIR=/path/to/private/frozen-prepared
export CUDA_HOME=/usr/local/cuda-13.0
export CXX=/usr/bin/g++-11
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

python3 scripts/check_environment.py
bash scripts/start.sh --gpu 0 --port 8000 --variant ngram3
```

只监听 `127.0.0.1`；API key 使用环境变量，不进入启动命令或公开数据。物理 GPU 由参数指定；启动前连续检查目标卡无其他计算进程且 GPU utilization≤3%，加载前 free memory≥28 GiB；占用时拒绝启动。`.local/service/receipt.json` 保存进程 UID、PID、start ticks、GPU UUID、端口与日志。停止只针对该 receipt 的进程会话。

包装脚本设置的关键运行环境：

```bash
VLLM_PLUGINS=''                     # 阻止已安装 Omni 插件覆盖原生模型注册
VLLM_USE_V2_MODEL_RUNNER=0           # 固定实测 V1 GPUModelRunner
VLLM_USE_FLASHINFER_SAMPLER=0
OMP_NUM_THREADS=1
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
```

核心入口为 `python -m vllm.entrypoints.openai.api_server`；完整参数来自 [configs/deploy.json](configs/deploy.json)。实际命令与审计写入 `.local/service/deployment/`。该版本入口有弃用提示，但这里保留实测入口；不能加 `--omni`。CUDA/toolchain、PYTHONPATH、scales 与审计环境应由包装脚本设置，不能只复制 speculative 参数。

## 实践：冻结输入 → 预热 → 测试 → 对照

按 [输入合同](data/readme.md) 准备原视频、全部参考图、最终 system prompt、shot plan/runtime context，并生成正确 SHA256。客户端不会重新切分镜头；新数据不能直接声称复现历史数字。

```bash
# 每轮预热最大输入样本 1 条，然后测试 manifest 全部 15 条；默认两轮
bash scripts/benchmark.sh --run-dir "$PWD/.local/n3-run" --rounds 2

# 查看汇总（含 Thinking）与每一条原始结果
cat .local/n3-run/pooled-results.json

# 只停本方案 receipt 对应服务
python3 scripts/stop.py

# 使用完全相同冻结输入测试原生无 n-gram 对照
bash scripts/start.sh --gpu 0 --port 8000 --variant control
bash scripts/benchmark.sh --run-dir "$PWD/.local/control-run" --rounds 2
python3 scripts/stop.py

# 可选：N5/K5，使用 configs/ngram5.json；历史结果未显示实用收益
bash scripts/start.sh --gpu 0 --port 8000 --variant ngram5
bash scripts/benchmark.sh --run-dir "$PWD/.local/n5-run" --rounds 2
```

每次使用新 run-dir，等待完整预热与正式结束；禁止给同端口发送其他生成请求。保存每条原始 SSE、event 时间戳、Thinking、完整 Caption JSON、token IDs、metrics、GPU 1 Hz 遥测、前后 `/metrics` 和 API 计数。`.local/` 全部忽略，**不能把私有运行产物直接提交到 GitHub**。包装脚本为发布时新增的路径参数化实现，只做静态审查，本次整理未再次跑 GPU。

## 计时与验收

- **条/分钟** = 完整正式请求数 / 正式批次 wall × 60；合并两轮使用总请求/总 wall，不能平均逐样本速度。
- **输出 TPS/卡** = Thinking + 最终 Caption 的 completion tokens / 总 wall / GPU 数。本方案 TP1，等于单卡实际输出吞吐；不是单请求解码速率。
- **TTFT**：HTTP 发出到首个 Thinking 或正文 token；**TPOT**：(请求 E2E−TTFT)/(completion tokens−1)。草拟验证可能一次返回多个 token，TPOT 是平均耗时而非每个相邻 SSE 到达间隔。
- **请求 E2E**：HTTP 开始至最终 SSE，含服务端排队、prefill、Thinking、Caption；**样本 E2E**另含客户端准备，不含线程池排队；**批次 wall**含全部客户端排队。
- 正式输入哈希、预检 token 数一致；无外来选卡进程；API 成功计数增量=15且服务未重启；观察到 Thinking；无错误/截断；token IDs/usage 匹配；流完整、JSON/基础结构完整。未通过者保留失败证据，不能算作完整任务。
- 如检查点、依赖源码、输入内容、采样、并发、KV scales、请求协议不同，单独记录新条件，不与历史数字直接归因比较。

## 公开边界

保留 90 条匿名指标及配置审计，未公开业务视频、参考图、system prompt、shot plan 内容、原始 Caption/Thinking、原数据来源、登录信息、主机 IP/GPU UUID 或凭证。15 条性能集不是带人工真值的精度集；本报告不声称音乐分类/H800 精度结果可迁移到该 Caption 实验。

[返回 RTX 5090 索引](../readme.md) · [原 Omni FP8 方案](../async-fp8-gemm-fp8-kv-tp1-c4/readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
