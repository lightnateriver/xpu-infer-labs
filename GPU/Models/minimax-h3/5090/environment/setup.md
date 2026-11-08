# 环境记录与重建入口

[observed.json](observed.json) 来自实测 Linux 容器的版本查询。CUDA 13.0 是 PyTorch 的构建版本，nvcc 13.4.92 是实验里为 JIT 提供的 Toolkit 编译器。版本清单用于对齐运行环境和排查偏差。

有匹配环境时，直接激活 Python 环境并从 [主 README](../readme.md) 生成 overlay。新建环境时可以用以下入口，安装源须提供实测版本的 CUDA wheel。

```bash
uv venv --python 3.12.16 .venv
source .venv/bin/activate
uv pip install --torch-backend=cu130 -c environment/constraints.txt \
  https://wheels.vllm.ai/5df63a7203cdb9328d824a27b682e77c7b9928be/vllm-0.32.0-cp38-abi3-manylinux_2_28_x86_64.whl \
  vllm-omni==0.32.0 flashinfer-python==0.7.2.post1
uv pip check
```

本次已在实测环境验证两份 overlay 的构建与源码哈希。上述新环境安装命令和外部 wheel 供应需要在接手机器上验证。

安装 FFmpeg/ffprobe 与 Git 后，检查 CUDA 和 FlashInfer：

```bash
python - <<'PY'
import torch
from importlib.metadata import version
from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_nvfp4

print("vLLM", version("vllm"), "Omni", version("vllm-omni"))
print("PyTorch", torch.__version__, "CUDA", torch.version.cuda,
      "cuDNN", torch.backends.cudnn.version(), "NCCL", torch.cuda.nccl.version())
for rank in range(torch.cuda.device_count()):
    print(rank, torch.cuda.get_device_name(rank), torch.cuda.get_device_capability(rank))
PY
nvcc --version
ffprobe -version
```

SP4 建议给 `FLASHINFER_WORKSPACE_BASE` 分配当前用户可写目录，在正式计时前完成 JIT 预热。V1 SP8 和原生 TP8 使用 8 张 RTX 5090，SP4 使用 4 张。确认选定卡无其他计算进程或显存占用。
