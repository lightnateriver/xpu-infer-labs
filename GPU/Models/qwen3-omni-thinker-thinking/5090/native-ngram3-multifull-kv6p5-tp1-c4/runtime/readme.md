# 实测运行代码

- `multigraph_worker.py`：26个额外精确query FULL Graph、持久ragged mask、真实decode资格判断、实际replay计数与可选前向逐元素检查。原生4个全4形状与不符合条件的回退路径保留。
- `native_worker.py`：原生vLLM Thinker和DeepStack modality metadata兼容修复。
- `mixed_fp8_worker.py`、`kv_audit_worker.py`、`caption_worker.py`：96个attention线性层FP8、KV/scales/精度和专家量化核对。
- `memory_worker.py`：诊断专用，等待Graph捕获完成再采样；空闲指令触发trim/reset/snapshot。正式速度与最终默认服务不启用该Worker。

以上文件与本轮实测源码按LF文本相同，见[evidence身份记录](../evidence/publication-provenance.json)。补丁仅进程内生效，安装包源码不写入；依赖固定的内部接口，不能假设升级vLLM仍可工作。额外FULL不是强行把ragged batch标成uniform，也不升级prefill。


[返回方案](../readme.md)
