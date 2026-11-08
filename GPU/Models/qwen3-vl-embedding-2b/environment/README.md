# 测试环境 · RTX 5090

单卡 32 GiB、x86_64 Linux；主机 264 逻辑 CPU，解码配置 8/32 进程，每进程 Torch=1。
Python 3.12.15、NVIDIA 驱动 580.126.09、CUDA runtime 12.8；torch 2.10.0 / torchvision 0.25.0，CUDA 12.8 用户态依赖；vLLM 0.19.0、Transformers 4.57.6、qwen-vl-utils 0.0.14、PyAV 17.1.0、NumPy 2.2.6、Pillow 13.0.0、imageio-ffmpeg 0.6.0。

[setup.sh](setup.sh) 安装固定 Python 依赖；需事先提供能支持 RTX 5090 的驱动与 CUDA PyTorch 环境。
安装后会改变依赖，请用独立环境。当前验证使用已有环境，本次没有从空白主机重建整个驱动/系统栈。
实际复测包版本与源代码 SHA256 记录在公开汇总中。PyAV 自带的 FFmpeg 行为也是输入数值合同的一部分。
