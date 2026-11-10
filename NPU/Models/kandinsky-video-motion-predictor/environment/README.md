# 环境与来源

GPU实测：Linux x86、Python3.11.5、torch2.10.0+cu128、torchvision0.25.0、CUDA12.8。
NPU实测：Linux ARM、Python3.12.13、torch2.10.0+cpu、torch-npu2.10.0.post4、CANN9.1；910B1 64GB。
权重参数保留原FP32存储，推理使用FP16 autocast，不是BF16或全FP32；分数输出转FP32，统计用FP64。
不覆盖已有设备环境，先按硬件供应商要求安装匹配的驱动/torch/torchvision/torch-npu/CANN。

通用依赖复用numpy、imageio[pyav]、av、huggingface-hub、safetensors、einops、transformers、timm1.0.15、easydict1.13。
本设备精确实测依赖见requirements-tested.txt；GPU/NPU的numpy、PyAV、transformers和Hub版本不同，未假装环境完全一致。
x86实测decord0.6.0；ARM无同版wheel时用decord2 3.4.0，导入名仍为decord。
不要同时安装decord和decord2；两套wheel在本批输入已核对像素100/100一致，不能外推所有编码。

```bash
python -m pip install -r environment/requirements-tested.txt
python scripts/fetch_model_source.py --destination "$UPSTREAM_DIR"
export MOTION_UPSTREAM_DIR="$UPSTREAM_DIR"
```

上游模型实现：[gen-ai-team/kandinsky-video-tools](https://github.com/gen-ai-team/kandinsky-video-tools)，
模型权重：[ai-forever/kandinsky-video-motion-predictor](https://huggingface.co/ai-forever/kandinsky-video-motion-predictor)。
`upstream.json`固定公开revision和8个模型文件SHA256，与实测源码逐一核对；本仓库不复制模型实现或权重。
下载脚本不下载视频/示例；自备对应上游使用权限。断网环境可预先准备这些文件并设置环境变量。

设置`QC_CKPT_DIR`为权重根目录，下面应有`ai-forever/kandinsky-video-motion-predictor/`，
含config.json、model.safetensors；校验值在upstream.json。没有该环境变量时模型加载明确报错。
CPU预处理不加载模型，不要求权重变量。

这里记录实测版本，不宣称在空白机器从零重装已验证；安装脚本不升级或替换硬件驱动栈。
