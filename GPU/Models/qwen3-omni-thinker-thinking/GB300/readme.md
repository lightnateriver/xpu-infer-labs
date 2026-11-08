# Qwen3-Omni Thinking · GB300 · TP1 / C56

历史已测的高吞吐方案：**15.358 条完整结果/分钟，折合 3.907 秒/条**。这是 56 并发下整批 wall time / 完整条数；**单请求中位等待 127.39 秒**，不是单条 3.9 秒响应。该目录记录 2026-10-20 的官方 Thinking 权重实验，未混入后续自训权重结果，也未在发布时重跑 GPU。

| 指标 | 历史结果 |
|---|---:|
| 硬件 | 单卡 NVIDIA GB300，284208 MiB，Linux aarch64 |
| 独立样本 / 请求 | 15 / 56（固定顺序重复 3–4 次） |
| 正式测量轮数 / 独立预热 | 1 / 1 |
| 完整有效 / 截断 / 失败 / 抢占 | 56 / 0 / 0 / 0 |
| 完整结果吞吐 | 15.3583125 条/分钟 |
| 整批 wall time | 218.774 秒 |
| 吞吐折算时间 | 3.90668 秒/完整条 |
| 单请求 HTTP E2E 中位数（含 Thinking） | 127.393 秒 |
| 输出 token/s/卡（含 Thinking） | 1379.414 |
| KV 使用峰值 | 60.32% |

结构和完整性通过，未进行人工语义质量打分。一次历史测量不足以证明跨任务、跨权重或长期稳定最优；“最优”指本次指定的已测方案。与 BF16 KV128 + output32768 历史记录相比，吞吐约 +14.1%，但同时改变 KV dtype 和输出预算，不能当作纯 FP8-KV 消融。

## 配置

官方 `Qwen/Qwen3-Omni-30B-A3B-Thinking` BF16 源权重，在线 `nvfp4_per_token` 量化 MoE experts；dense attention、router、lm_head、visual/audio 编码器保留 BF16。FP8 E4M3 KV 为 128 GiB；不使用 5090 的 attention FP8 或 N-gram 方案。仅 Thinker 文本输出，不启动 Talker。

TP1，C56，context65536，chunk65536；compile mode3，`FULL_DECODE_ONLY`，capture sizes `[1,2,4,8,16,32,56]`；async 关闭。prefix cache 与多模态 processor cache 关闭，媒体 uuid 每请求独立。attention/MoE backend 保持 auto。精确参数见 [configs/deploy.json](configs/deploy.json)，补丁见 [runtime](runtime/readme.md)。

请求开启 `enable_thinking=true`，temperature0、seed0，文本流式输出；视频 2 FPS、`num_frames=-1`，视频音频开启，保留全部参考图。输出上限按 `min(40960, 65536 - 实际输入 tokens)`，保留完整输入和原生 context，不扩大上下文。

## 复现服务

Linux ARM64 + NVIDIA Docker，单卡显存须容纳权重、128 GiB KV 和编译 workspace。自备合法模型、冻结输入和相同环境。

```bash
export MODEL_DIR=/path/to/Qwen3-Omni-30B-A3B-Thinking
export VLLM_IMAGE=sha256:bbe7045055707d1027079ed01cca40812578a235c56032d85c16108ad473384c
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"
python3 scripts/serve.py --gpu 0 --port 8000
# 待服务 ready，准备本地数据并测一次；预热与 tokenize 均排除在正式整批计时外
python3 scripts/benchmark.py --input /path/to/private-input.json --out .local/run
python3 scripts/serve.py --stop
```

image ID 是实测本地镜像内容标识，不能直接 `docker pull sha256:...`。需预先导入相同镜像，或设置其他镜像并自行重新资格验证；只匹配 vLLM 版本号不足以保证一致。启动器强制校验两个上游原文件 SHA256；不符合时拒绝挂载补丁。启动不会停止已有 GPU 进程；仅停止自己 receipt 对应的容器。

公开启动器/客户端是参数化移植包装，本次做 CPU/mock 验证，未重新 GPU 实测；不是内部控制器逐字发布。不要把用自备输入跑出的新数值声称为原历史数据的精确复现。

## 输入及计时合同

输入 JSON 格式见 [data/readme.md](data/readme.md)。历史 15 个独立视频约 12–15 秒，参考图 3–5 张，固定逐样本 system prompt；56 请求是同一冻结集的重复并发，不是 56 个独立视频。私有 prompts、媒体、业务输出和 CoT 不发布，匿名指标见 [evidence/requests.json](evidence/requests.json)。

本地客户端保存每条原始 SSE、最终输出、Thinking、预算和指标，绝不自动重试；默认校验 stop、非空 Thinking、JSON 输出和 token 预算。业务 schema 需自行提供 validator；这与历史内部完整 schema 校验有所区别。预热必须是独立样本，不得用正式输入预热。

历史批次计时包含媒体读入、Base64、排队、HTTP 全流及结果落盘，排除部署/编译、tokenize preflight 和独立预热。单请求 HTTP E2E 不含媒体准备。首次 token 时间不是完整 caption 时间，单条延迟也不能除以并发数冒充实测吞吐。缓存条件为文件系统和编译缓存热、无请求 KV/输出复用。

## 证据及验证

- [匿名汇总](evidence/summary.json)、[56 条匿名指标](evidence/requests.json)、[独立 HTML 报告](report.html)。token/s 包含推理 tokens。
- `python3 scripts/verify.py` 检查 56 条、完整性、预算及吞吐一致性，以及公开 runtime SHA256。
- 正式运行需独占目标 GPU，并持续检查计算 PID、抢占、dtype 和 Graph 调用；历史记录 56/56 隔离有效，Graph measured FULL replay=18018，NONE=56（含 prefill）。
- 公开包删除登录信息、节点/卡身份、私有路径、原样本 ID、prompt/媒体内容、caption/CoT 及私有溯源哈希。原始证据保留在私有归档，不随仓库发布。
