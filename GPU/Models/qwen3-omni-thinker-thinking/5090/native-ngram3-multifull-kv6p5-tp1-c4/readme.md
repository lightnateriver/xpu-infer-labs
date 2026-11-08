# RTX 5090 · N3/K3 + 多 FULL Graph + KV 6.5 GiB

**本轮采用：TP1 / 客户端C4 / 服务max_num_seqs=4 / FP8 KV 6.5GiB / chunk4096。**

2026-10-21 冻结15条Caption输入，最优方案三轮 **45/45完整有效、零抢占**，有效吞吐 **3.574899条/分钟**，较本轮C4基线 **+10.33%**。完整轮的输出TPS/卡中位数 **310.22**，TTFT P50 **6.912s**、TPOT P50 **10.163ms**、处理E2E P50 **62.275s**。所有Thinking时间与输出tokens计入。

## 七项完整对照

| 方案 | 有效/总请求 | 有效条/min | 较本轮C4 | 完整轮TPS/卡中位数 | 抢占次数 |
|---|---:|---:|---:|---:|---:|
| C4 / KV6.25GiB | 59/60 | 3.240227 | — | 296.58 | 10 |
| C3 / KV6.25GiB | 45/45 | 3.075830 | −5.07% | 274.17 | 0 |
| Token预算90% | 45/45 | 2.931049 | −9.54% | 265.32 | 0 |
| Token预算95% | 45/45 | 3.157242 | −2.56% | 280.21 | 0 |
| **C4 / KV6.5GiB** | **45/45** | **3.574899** | **+10.33%** | **310.22** | **0** |
| C4 / KV6.75GiB | 45/45 | 3.395806 | +4.80% | 308.23 | 0 |
| KV6.5GiB + chunk2048 | 45/45 | 3.315226 | +2.31% | 305.57 | 2 |

共22轮、330条正式请求，每方案至少3轮完整结果。C4额外一轮含1个`sample_id`不匹配，正常停止、无截断；没有修复输出或丢弃该轮耗时。主指标=sum(有效Caption)/sum(全部正式尝试wall)×60。其余速度统计仅取完整轮中位数，不能将median TPS当作全部尝试的token总量/总wall。

chunk2048的+2.31%是相对C4基线，**相对其KV6.5GiB父方案约−7.26%**。零抢占与吞吐分别判断；本轮采用增加KV而非限制C4。KV6.5GiB峰值仍为99.10%，不保证更长输入或Thinking也零抢占。3轮不足以证明统计显著性或通用生产收益。

## 目录

| 目录 | 内容 |
|---|---|
| [configs](configs/readme.md) | 最优部署、六个对照、独立显存诊断配置、实验顺序和固定scales |
| [runtime](runtime/readme.md) | 实测多FULL Worker、原生多模态兼容、FP8/KV审计、独立显存Worker |
| [benchmark](benchmark/readme.md) | 实测自然Caption、固定长度预热、FIFO token准入和SSE计时 |
| [scripts](scripts/readme.md) | 参数化启动/停止、环境/源码核对、重复测试及汇总 |
| [data](data/readme.md) | 私有冻结输入合同和匿名manifest示例 |
| [environment](environment/readme.md) | 版本及关键源码SHA256，不能只凭版本号宣称一致 |
| [evidence](evidence/readme.md) | 330条匿名指标、22轮结果、KV轨迹、准入及Graph检查 |
| [report](report/readme.md) | 独立红色浅色HTML、柱状图、时间线和可重建生成脚本 |

## 保留的计算条件

权重为非官方 **Qwen3-Omni-30B-A3B-Thinking-heretic-NVFP4**。96个attention QKV/O线性层FP8 E4M3 W8A8，静态tensorwise权重、动态activation、BF16输出；48层MoE保留NVFP4，router/lm_head/多模态编码器保持原精度。48层FP8 KV、固定K/V scales。不是官方BF16全模型，也不是额外部署FP8 drafter小模型。

原生vLLM V1、TP1、Async、GPU n-gram N3/K3、FlashInfer XQA BF16 Q。FULL_AND_PIECEWISE、capture sizes=[1,2,4,8,16]；最多4请求的query∈{1,4}额外26张FULL，原4个全4形状继续原路径。只有真正decode升级，prefill/encoder、其他query及不满足条件时保留回退。固定ragged mask，按精确query元组分派；该方案包含Graph、形状和mask缓存的组合，不能称为纯Graph消融。prepare融合补丁未开启。

## 启动最优服务

在Linux RTX5090 GPU主机、已匹配的Python3.10环境中进入本目录：

```bash
export MODEL_DIR=/path/to/model
export DATA_DIR=/path/to/private/dataset
export PREPARED_DIR=/path/to/private/frozen-prepared
export CUDA_HOME=/usr/local/cuda-13.0
export CXX=/usr/bin/g++-11
export VLLM_API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(32))')"

python3 scripts/check_environment.py
bash scripts/start.sh --gpu 0 --port 8000 --variant kv6p5
bash scripts/benchmark.sh --run-dir "$PWD/.local/kv6p5-run" --rounds 3
cat .local/kv6p5-run/pooled-results.json
python3 scripts/stop.py
```

GPU编号为调用方指定的物理卡；API仅监听127.0.0.1。连续检查选卡无其他计算进程、利用率≤3%，不自动终止别人的服务。加载前核对模型指纹、版本、关键源码；启动后核对96层FP8、48层FP8 KV、原生模型/调度器/Proposer与26张额外FULL。安装的Omni插件被显式禁用；核心入口是`vllm.entrypoints.openai.api_server`，不能加`--omni`。

`configs/deploy.json`为最优配置。KV预算精确为 **6,979,321,856 bytes**，对应 **141,984 tokens**；显式字节预算优先于显存比例。启动脚本把模型路径、别名和端口替换为调用方设置，其余计算参数保留。新包装脚本只在本次发布做静态审阅，未重新进行GPU部署测试；实测Worker和请求生成代码按原文保留。

## 对照顺序与显存诊断

历史初始顺序是C4,C3,C3,C4,C4,C3，C4含额外补测；准入按90/95交错，随后独立显存诊断、6.5/6.75GiB和chunk2048。详见[实验计划](configs/experiment-plan.json)。公开包装脚本是每方案连续重复的可移植实现，不是原内部交错控制器的逐时复刻；需要同等严格交错时可用[底层客户端](benchmark/readme.md)按计划逐轮调度，记录额外预热条件。

各方案独立选用以下variant，停止当前receipt所属服务后再启动，使用新的run-dir：

```bash
# 本轮基线；同样保留多FULL，KV6.25GiB
bash scripts/start.sh --gpu 0 --port 8000 --variant c4
bash scripts/benchmark.sh --run-dir "$PWD/.local/c4-run" --rounds 3
python3 scripts/stop.py

# c3、budget90、budget95同样操作；server max_num_seqs仍为4
# 在扩大KV前，先启动独立诊断Worker
bash scripts/start.sh --gpu 0 --port 8000 --variant memory
bash scripts/benchmark.sh --run-dir "$PWD/.local/memory-run"
cat .local/memory-run/capacity-decision.json
python3 scripts/stop.py

# 仅当本机诊断eligible_kv_gib包含目标值，才尝试kv6p5或kv6p75
bash scripts/start.sh --gpu 0 --port 8000 --variant kv6p5
bash scripts/benchmark.sh --run-dir "$PWD/.local/kv6p5-repeat" --rounds 3
python3 scripts/stop.py

# chunk2048保持KV6.5GiB和C4，用于比较prefill chunk
bash scripts/start.sh --gpu 0 --port 8000 --variant chunk2048
bash scripts/benchmark.sh --run-dir "$PWD/.local/chunk2048-run" --rounds 3
```

显存Worker仅用于诊断，不计正式速度。确认API和选卡归属空闲后empty_cache/reset峰值，运行15条自然请求，核算活跃峰值、非Torch占用、Graph池闲置和512MiB余量；不直接删除Graph池。容量资格不是OOM保证，扩大后仍要通过启动、固定预热与完整自然请求；失败时保存证据，手动用receipt停止自己的服务后回到c4，不继续盲目扩容。

准入为FIFO，预算=实际KV tokens×90%/95%，每条预留prompt tokens+9000输出tokens，线程数最多4。9000来自历史输出长度的经验，不是最大输出保证；实际Thinking超过预留仍可能抢占。未修改请求的24576输出上限。

## 输入、Thinking与验收

15条冻结视频+原音轨+全部64张参考图，长system prompt+ShotPlan/runtime context，2fps，共419460输入tokens/轮；单视频约12–15秒。内容不公开，调用方自备有权使用的输入。[manifest示例](data/manifest.example.json)仅说明字段，不是业务样本或完整性能集。

温度0、seed0、Thinking开启、max_tokens=24576、context=65536。启动后先对15条做token预检，再4条×4096固定输出预热及Graph数值审计，排除在排名外；自然输出至少3轮完整，必要时补测，最多9次。显存方案只做一轮15条独立诊断，不计排名。原历史起点的自然单条预热及其继承的服务热态已记录；发布包装脚本统一固定预热，因此不承诺复刻相同耗时。

保留每条原始SSE、Thinking、Caption、token IDs、timing、hash、GPU1Hz遥测与前后API计数至忽略目录`.local/`。无效自然输出也完成API计数核对并保留，再补测；所有正式尝试时间进入有效吞吐。传输、GPU隔离或计数失败直接中止，不静默当作可接受的生成随机性。

TTFT从HTTP发出到首个Thinking/正文片段；TPOT=(请求E2E−TTFT)/(completion_tokens−1)。处理E2E包括媒体准备但排除token预检和客户端排队；提交到结束E2E包含线程池/token预算等待；批次wall包含全部正式排队。输出TPS含Thinking和正文，TP1直接等于每卡值。自然长度仍会变化，不能全归因为执行效率变化。

最后一次运行检查45次逐元素一致、覆盖24种遇到的额外query形状；启动总计26张额外FULL，两者分别记录。**这不是人工Caption质量评估。** 15条没有语义真值，不能据此断言FP8无损、与BF16/H800等精度，或音频分类精度迁移。

## 发布边界

公开的是配置、关键源码、匿名指标、审计及独立报告。不含登录/机器信息、原来源、私有路径、媒体、prompt、Caption或Thinking正文。scales仅保留数值字段；模型指纹沿用历史记录，本轮没有重算所有权重SHA。新包装脚本与原运行环境的字节一致性边界见[环境说明](environment/readme.md)。

[完整性能报告](report/report.html) · [RTX5090索引](../readme.md) · [历史N3方案](../native-ngram3-async-fp8-tp1-c4/readme.md)
