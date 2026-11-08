# 测试环境 · Ascend 910B1

| 项目 | 实测版本 |
|---|---|
| 系统 / CPU | aarch64 Linux / 192 逻辑 CPU |
| 加速卡 | 单张 Ascend 910B1，64 GiB |
| 驱动 / CANN | 25.5.1 / 9.1.0 |
| Python | 3.12 |
| torch / torch-npu / torchvision | 2.10.0+cpu / 2.10.0.post4 / 0.25.0+cpu |
| vLLM | 0.30.0+empty，源码 `ced6857afa0ea7b2e3f0846a62e1394e90f15607` |
| vLLM-Ascend | 0.19.1rc2.dev2331+g16074608c，源码 `16074608c71ef5adce4e2a4e475848b8fc037297` |
| Transformers / qwen-vl-utils | 5.14.1 / 0.0.14 |
| PyAV / NumPy / Pillow / imageio-ffmpeg | 17.1.0 / 2.2.6 / 13.0.0 / 0.6.0 |
| RGB 编译器 | g++ 11.4.0，C++17 -O3 -fPIC -shared |

本次检查两个后端源码目录的 Git 工作树干净。精度/图适配作为交接目录中的 runtime 补丁动态安装，不直接改后端源码。
测试使用已有 CANN 容器环境；不提供私有镜像、地址或镜像仓库访问凭证。本次验证的是交接代码在该环境中运行，没有从空白系统验证一份 Dockerfile。

交接者应先匹配 CANN/驱动和 ARM torch-npu，再从上述精确提交构建 vLLM 的 empty 后端与 vLLM-Ascend。不要根据显示版本的不同前缀推断这两个包来自相同 release tag；以提交为准。
使用所选提交中的构建说明与依赖，完成后核对表中元数据；本 runtime 会拒绝关键包版本漂移。
`ascend` platform plugin 与 `ascend_model` model plugin 均来自 vllm_ascend。

```bash
python -m pip install 'transformers==5.14.1' 'qwen-vl-utils==0.0.14' \
  'av==17.1.0' 'numpy==2.2.6' 'pillow==13.0.0' 'imageio-ffmpeg==0.6.0'
python runtime/qwen_embedding/build_rgb_core.py
python tests/verify_cpu.py
```

RGB 源码 SHA256 `5fa4b1923e0d6f268223bc04199a37bf8aaf2858a9b94f470b70aa2c30ea25ab`。
本次 ARM 重编译二进制 SHA256 `e066ce85075e562cda1620bdc536ac6be45f18a7774c3f3a6e32a1ee71f8c8e0`；其他编译器的二进制哈希可不同，应运行精确像素测试。
自动 NUMA 绑定发生在引擎初始化后，保留其运行状态；不要照搬初始化阶段的 CPU affinity 快照手工放宽绑定。
