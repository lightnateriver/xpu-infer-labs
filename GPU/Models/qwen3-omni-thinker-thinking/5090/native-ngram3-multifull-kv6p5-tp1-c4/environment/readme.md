# 环境与复现边界

`versions.json`和`reference-versions.txt`给出已记录的Python3.10、vLLM0.30.0、PyTorch2.13.0+cu130、FlashInfer0.6.18.post1等依赖；CUDA Toolkit13.0/nvcc13.0.48、GCC11.4.0、ffmpeg/ffprobe。SM120 RTX5090，单卡TP1。

`package-source-manifest.json`合并历史原生环境身份和本轮4个关键安装源码的SHA256，路径相对于Python site-packages。开始/结束4个关键源码完全相同，未进行安装包写入。原环境存在已有兼容/编译修改，不宣称pip安装同版本即可一致；不提供私有镜像，也不从版本号推断可复现。check_environment.py在发布包装脚本中对版本和源码指纹做只读检查，失配需先定位差异，不能跳过后宣称匹配环境。

禁用Omni插件和V2 runner、保留Async、XQA BF16 Q/FP8 KV、固定scales；模型指纹沿用先前记录，本轮未重算所有权重的SHA。新包装脚本未做本次GPU复测；现有实测及静态发布检查分别记录。


[返回方案](../readme.md)
