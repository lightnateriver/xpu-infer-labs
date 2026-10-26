# 固定 runtime

保留历史实测的 vLLM 0.31.0 Thinker 与 CUDA Graph 文件及审计 helper，启动时只读挂载；receipt 内记录原文件与补丁 SHA256。

Thinker 兼容补丁在 torch.split 后恢复 modality 标记；其余增加首次 forward 的权重 dtype 元数据审计及 native V2 Graph 调用计数。审计不改变 tensor 数值，不添加 CUDA 同步。Graph NONE 计数包含 prefill，不能理解成错误数。

启动后可用 `docker exec CONTAINER sh -c 'cat /dev/shm/codex-omni-weight-audit-*.json'` 检查 packed NVFP4 experts、float8 group scales、FP32 global scales，以及受保护组 BF16。Graph 审计文件同在容器 `/dev/shm`；新进程命名空间隔离旧计数。API health 不能代替这些核对。

上游衍生文件保留 Apache-2.0 版权头，见 [THIRD_PARTY.md](../THIRD_PARTY.md)。
