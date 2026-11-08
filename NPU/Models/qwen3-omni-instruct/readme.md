# Qwen3-Omni Instruct · Ascend 910B3 单卡

本工作负载的最快实测单卡配置为 **TP1 W8A8、关闭 EP、BF16 KV 20 GiB、客户端并发 4**。使用官方 Qwen3-Omni-30B-A3B-Instruct，关闭 thinking，仅部署 Thinker 文本输出。

| 并发 | 正常结束 | 批次耗时 | 请求 P50 E2E | P50 TTFT | 正常完成条/分钟/卡 | 输出 token/s/卡 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 15/15 | 629.001 s | 39.681 s | 10.794 s | 1.431 | 44.68 |
| 4 | 15/15 | 304.357 s | 70.040 s | 12.658 s | 2.957 | 83.59 |

每个配置测一轮同样的 15 条真实多模态样本，随机预热一条，预热与预校验不计时。C4 批次吞吐约为 C1 的 2.07 倍。两轮零截断、零抢占，直接 JSON 均为 0/15，去掉完整 Markdown 围栏后正文结构均为 15/15。正常结束与语义准确分别验收。历史 C4 KV12→KV20 测到 2.83→2.92 条/分钟，抢占 1→0；历史 KV20 C8 为 2.89 条/分钟且有一次抢占。单轮结果不构成长期 SLA。匿名原始指标见 [performance.json](performance.json)。

## 1. 准备权重

从 [Qwen 官方仓库](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) 下载完整 BF16 checkpoint。使用 [Ascend msModelSlim](https://gitcode.com/Ascend/msmodelslim) 中包含 Omni W8A8 适配的 26.1.0 源码，在支持 910B 的 CANN/torch-npu 环境中按上游安装说明安装。实测量化使用独立环境的 Transformers 4.57.3，确保 FFmpeg、PyAV、librosa 和 audioread 能解码校准素材。推理环境保持镜像自带依赖。

核对工具自动选中的 Omni W8A8 配方与 [quantization.yaml](quantization.yaml) 一致。它采用动态 per-token INT8 activation、per-channel INT8 weight，排除 audio tower，使用工具内置 `calibVideos`。若使用其他校准集，应记录版本和模态，重新评估输出质量。

```bash
bash quantize.sh "$BF16_DIR" "$W8A8_DIR"
```

输出需包含 `quant_model_description.json` 和 safetensors 分片。[weight-manifest.json](weight-manifest.json) 是历史产物的大小和 SHA256，可在已有该 checkpoint 时逐文件核对。历史量化源码归档的 SHA256 为 `baf8e5dccd732b6efb5dc0ec6df82fb1dd7515190a043428ccc5337ce860c228`。归档未公开，只有版本和配方可用时，重建权重需以实际输出验收，无法保证字节相同。

## 2. 构建与运行

在 Linux Ascend 910B3 64 GiB 主机安装匹配容器的驱动和固件。基座为 `quay.io/ascend/vllm-ascend:v0.30.0`，历史本地镜像 ID 为 `sha256:74224d25260decde6abdd26c6ebc383faadbaf792115631bdbe966793742c4db`。记录过的运行环境为 vLLM 0.30.0+empty、vLLM Ascend 0.30.0、torch 2.10.0+cpu、torch-npu 2.10.0.post7、Transformers 5.14.1。Dockerfile 按顺序应用 modality、input-alignment、MRoPE 补丁，镜像不匹配时构建会报错。

```bash
docker build -t omni-instruct-npu:w8a8 .
NPU_DEVICE=0
docker run --rm --network host \
  --device "/dev/davinci${NPU_DEVICE}" --device /dev/davinci_manager \
  --device /dev/devmm_svm --device /dev/hisi_hdc \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro \
  -v "${W8A8_DIR}:/model:ro" \
  -e "ASCEND_RT_VISIBLE_DEVICES=${NPU_DEVICE}" \
  --shm-size 16g omni-instruct-npu:w8a8
curl -f http://127.0.0.1:18096/health
```

根据主机设备节点和容器内设备映射调整 device 与可见卡号。启动参数见 [serve.sh](serve.sh)，包括 max context 65536、batched tokens 4096、max seqs 8、异步调度和 FULL_DECODE_ONLY 图捕获 [1,2,4,8]。服务只监听本机，按需改端口。

## 3. 复跑与对照

准备 15 行 JSONL，每行是一个完整 OpenAI chat completions 请求体。保持 System Prompt、图片与视频顺序、素材字节和模板不变。媒体可用 data URL，或由模型服务器可访问的 HTTP URL。模型名为 `qwen3-omni-instruct`，历史参数为 temperature 0、top_p 1、top_k -1、repetition_penalty 1、max_tokens 24576，`media_io_kwargs.video = {"fps":2,"num_frames":-1}`，`mm_processor_kwargs = {"use_audio_in_video":true}`。公开包没有业务样本与 prompt，需使用有权限的素材。历史严格对齐还比对了实际 prompt token IDs、前端张量指纹和 MRoPE。下面的通用客户端测性能和原始 JSON，可另做输入指纹审计。

```bash
python3 benchmark.py requests.jsonl --concurrency 1 --output c1.json
python3 benchmark.py requests.jsonl --concurrency 4 --output c4.json
```

输出包含逐条原文、TTFT、请求 E2E、结束原因和用量。正式计时覆盖整批 HTTP 请求。对比时固定请求集合和输出预算，核对实际输入、输出长度和截断。发布时使用历史 NPU 实测数据，公开构建和压测脚本没有在 NPU 上重新执行。
