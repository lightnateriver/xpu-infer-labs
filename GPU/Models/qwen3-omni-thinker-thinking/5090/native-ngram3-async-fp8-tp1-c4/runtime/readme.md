# 实测原生 Worker 与兼容修复

调用链：`NativeAuditWorker → MixedFp8AuditWorker → KVAuditWorker → CaptionBaselineWorker → vllm.v1.worker.gpu_worker.Worker`。四个文件保持实测源码逻辑与文本（规范 LF）一致；不使用 Omni GPUARWorker。

1. `caption_worker.py`：移除检查点 compressed-tensors 中过宽的语言层 ignore 规则，防止 NVFP4 experts 误变 BF16；逐专家审计。
2. `kv_audit_worker.py`：检查选卡身份/空闲内存、加载冻结 scales、48 层 FP8 KV 实际分配及 FlashInfer XQA/BF16 Q 路由。
3. `mixed_fp8_worker.py`：要求恰好 96 个目标 FP8 线性层、48 BF16 router，实际 CutlassFP8ScaledMMLinearKernel；拒绝意外 FP8 层。
4. `native_worker.py`：修复原生 Qwen3-Omni DeepStack 切分视觉 embedding 后丢失 modality metadata。仅给替换后的 tensor 复制既有 metadata，不修改数值；同时审计 native 模型类、V1 runner、async 与 GPU proposer。

兼容补丁定位 `embed_input_ids` 中恰好一个赋值位置并记录修改前后源码 SHA；不匹配即失败，不能静默套用新版 vLLM。补丁只在该 worker 进程内生效，不修改共享 site-packages。代码针对已记录版本，不能视为所有版本通用插件。

[配置](../configs/readme.md) · [方案与运行方式](../readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
