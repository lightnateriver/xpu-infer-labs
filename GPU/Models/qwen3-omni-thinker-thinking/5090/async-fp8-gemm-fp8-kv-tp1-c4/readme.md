# RTX 5090 · 异步调度 + FP8 GEMM + FP8 KV

**历史实测吞吐：3.024 条/分钟，238.54 输出 token/s/卡。**

RTX 5090 32 GiB × 1，TP1，客户端 C4。2026-10-20 两轮测试，每轮完整预热 1 条，再正式测试 15 条；30/30 协议完整、无截断，Thinking 全程计入。此方案实际包含 FP8 KV；不是刚测的 BF16 KV 变体。

## 目录

| 目录 | 内容 |
|---|---|
| [configs](configs/readme.md) | 实际部署 JSON、原 K/V scales、模型 SHA256 指纹 |
| [runtime](runtime/readme.md) | 原始量化/多模态兼容修复及实际加载审计 worker |
| [scripts](scripts/readme.md) | 参数化启动、停止、环境检查、测试及汇总脚本 |
| [benchmark](benchmark/readme.md) | 流式 Caption 请求、Thinking/SSE 解析、计时与隔离检查 |
| [environment](environment/readme.md) | 实测版本记录；不是完整二进制环境锁文件 |
| [data](data/readme.md) | 自备冻结输入的目录/manifest 合同；无业务媒体或 prompt |
| [evidence](evidence/readme.md) | 匿名逐样本指标、合并结果、实际 kernel/KV 审计 |
| [report](report/readme.md) | 独立 HTML 性能报告及可重复生成代码 |

## 性能指标

| 指标 | 两轮合并 |
|---|---:|
| 完整正式请求 | 30/30（15 个不同输入 × 2 轮） |
| 完整任务吞吐 | **3.023557 条/分钟** |
| 输出吞吐 | **238.535153 token/s/卡** |
| 正式 wall 合计 | 595.325251 s |
| 输出 tokens（含 Thinking） | 142,006 |
| 输入 tokens | 838,920 |
| TTFT P50 | 7.656512 s |
| TPOT P50 | 13.688798 ms |
| 请求 E2E P50（含 Thinking） | 68.863130 s |
| 样本 E2E P50（另含客户端准备） | 70.179204 s |

3.024 是已保存的历史观测，不是吞吐保证。同配置另一次两轮复测为 **2.815 条/分钟、249.75 token/s/卡**；输出长度变化会同时影响条/分钟和 token/s。固定 6.25 GiB 预算改 BF16 KV 的后续实验为 **1.680 条/分钟、154.42 token/s/卡**，该变体不属于本目录默认配置。

## 卡型与环境

- RTX 5090 32 GiB，SM120；Linux x86_64；Python 3.10；原生 venv。
- **vLLM 0.30.0 / vLLM-Omni 0.30.0**。
- PyTorch 2.13.0+cu130；Transformers 5.14.1；compressed-tensors 0.17.0；FlashInfer 0.6.18.post1；Triton 3.7.1。
- CUDA Toolkit 13.0（nvcc 13.0.48）和 GCC 11.4.0；需要完整 JIT 编译工具链以及 ffmpeg/ffprobe。
- 实验权重：`Qwen3-Omni-30B-A3B-Thinking-heretic-NVFP4`。它是非官方衍生权重，**不是官方 BF16 Thinking 模型**。启动默认检查随包 10 个文件的 SHA256。

原环境没有已保存的可验证包构建 commit，也没有随包公开的 Docker 镜像。`environment/reference-versions.txt` 记录实际主依赖版本，不宣称在任意干净机器上执行 pip 即可得到字节一致环境。应使用等价的已验证环境或匹配的 wheel 集合，再执行环境检查。

## 启动方式与命令

在 Linux GPU 主机上进入本目录，并激活匹配的 Python 3.10 环境。自备有权限使用的同指纹模型与冻结输入：

```bash
export MODEL_DIR=/path/to/model
export DATA_DIR=/path/to/private/dataset
export PREPARED_DIR=/path/to/private/frozen-prepared
export CUDA_HOME=/usr/local/cuda-13.0
export CXX=/usr/bin/g++-11
# 生成本地 API key；不要使用 GitHub token 作为推理服务 key。
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

python3 scripts/check_environment.py
bash scripts/start.sh --gpu 0 --port 48121 --model-dir "$MODEL_DIR"
```

启动会检查物理卡型号、空闲状态、端口、模型指纹、实际 96 层 FP8 kernel 与 48 层 FP8 KV。卡被占用时直接拒绝，不停止其他作业。服务只监听 `127.0.0.1`，进程 receipt、日志、编译缓存位于默认忽略的 `.local/service/`。保持当前 shell 的 API key 供测试使用。

等效服务命令的核心为：

```bash
python3 scripts/api_compat.py serve "$MODEL_DIR" --omni \
  --deploy-config .local/service/deployment/deploy.json \
  --trust-remote-code --host 127.0.0.1 --port 48121 \
  --served-model-name qwen3-omni-thinking
```

应通过 `start.sh` 启动以同时设置 worker 的审计环境、原 K/V scales、CUDA/编译器路径和物理 GPU 映射；仅复制核心命令会遗漏这些设置。

## 测试方案与命令

```bash
bash scripts/benchmark.sh \
  --data-dir "$DATA_DIR" --prepared-dir "$PREPARED_DIR" \
  --run-dir .local/my-caption-run --concurrency 4 --rounds 2

# 测试脚本已自动汇总，也可手动执行：
python3 scripts/summarize.py --run-dir .local/my-caption-run --rounds 2

# 用完后只停止当前 receipt 所属服务：
python3 scripts/stop.py
```

`--run-dir` 必须为新目录。先对全部输入做 token 预检，再选输入 token 数最大的样本作完整预热，每轮正式 C4。输入为 15 个冻结样本，3 组各 5 条；原视频及同步音轨、全部 3–5 张参考图、2 fps、完整 system prompt 与既定 shot plan/runtime context。每轮输入合计 419,460 tokens。`enable_thinking=true`、`temperature=0`、`seed=0`、输出上限 16,384、上下文 65,536。

测试会保存每条原始输出、完整 Thinking、token IDs、SSE、GPU 监控和计时。它们保存在本地忽略目录，不会自动提交或上传。公开仓库不含原视频、原 prompt 或业务输出；需要有权限的原冻结输入才可数值复现。新数据可按相同合同运行，但不能直接对照 3.024。

### 指标口径

- 条/分钟 = 完整正式请求数 / 批次 wall × 60；合并两轮用总数 / 总 wall。
- 输出 token/s/卡 = 全部 completion tokens / 批次 wall / GPU 数，包含 Thinking 与 Caption。
- TTFT 从发出 HTTP 到首个文本 token（包括 Thinking）。
- TPOT =（请求 E2E − TTFT）/（输出 token 数 − 1），包含首 token 后调度、重算和流结束影响。
- 请求 E2E 从 HTTP 开始到 SSE 完成，包含服务端排队、prefill、Thinking、Caption；样本 E2E 另含客户端媒体准备；批次 wall 再包含客户端排队。
- 预热、启动/编译不计入正式 wall；P50 从全部 30 个请求重算；错误、截断或未通过输入/隔离/结构检查的请求不算完整输出。

## 优化手段

| 项目 | 本方案设置 |
|---|---|
| FP8 GEMM | 48 QKV + 48 O 共 96 层，E4M3 W8A8，静态 tensorwise 权重 scale + 动态激活量化，BF16 输出；实际 `CutlassFP8ScaledMMLinearKernel` |
| MoE / 其他层 | 专家保持原 NVFP4 CUTLASS；router、lm_head、视觉/音频编码器保留原精度 |
| 异步调度 | `async_scheduling=true`，实际 `OmniARAsyncScheduler`；`async_chunk=false` |
| CUDA Graph / 编译 | `enforce_eager=false`，mode=3，`FULL_AND_PIECEWISE`，capture `[1,2,4]`，`+rotary_embedding` |
| Chunked prefill | 开启，`max_num_batched_tokens=4096` |
| 并发 | `max_num_seqs=4`，客户端 C4，TP1 |
| KV Cache | FP8 E4M3，6.25 GiB，原独立校准 K/V scales |
| Attention | FlashInfer / XQA，Q 保持 BF16 |
| 输入缓存 | prefix cache 关闭，多模态处理器缓存 0，独立请求媒体命名空间 |

本目录展示组合实测；没有逐项独立消融收益，不将组合结果归因给某单项，不叠加理论加速。

## 报告、精度与公开边界

- [独立 HTML 性能报告](report/report.html)：下载后用浏览器打开；GitHub 文件页通常显示源码。
- [匿名完整数值](evidence/results.json) / [逐条 CSV](evidence/sample-metrics.csv)。
- 重建公开报告：`python3 report/build_report.py`。
- JSON 结构正确、无截断与 Thinking 完整不等于 Caption 语义正确。**尚无已验收的人工语义精度结论，不宣称精度无损**。
- 原核心 worker 与流式客户端来自实测代码；本目录新增路径参数化启动/测试封装，辅助模块移除了未使用的 Docker 后端，计时/请求算法保持不变。发布整理期间没有额外跑 GPU 实验。见 [代码适配记录](evidence/client-provenance.json)。
- 不含机器地址、账号、密码、私有绝对路径、GPU UUID、PID、私有镜像来源、SSH key、API key、GitHub token 或原业务数据。权重、运行环境和数据需自行准备并遵守各自许可。

## 新增原生 n-gram 方案

[原生 vLLM N3/K3 · TP1/C4](../native-ngram3-async-fp8-tp1-c4/readme.md)：2026-10-21 两轮观测 **3.161 条/分钟、273.71 token/s/卡**，Thinking 计入；与原生无 n-gram 的对照及 N5 试验见新报告。该数值不是上文 Omni 配置的复测，也未证明稳定吞吐或语义精度优势。

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
