# 配置

- `deploy.json`：默认kv6p5，KV=6979321856 bytes、chunk4096、TP1、server max_num_seqs=4。
- `c4.json`、`c3.json`、`budget90.json`、`budget95.json`：计算配置相同（KV6.25GiB）；客户端策略由`experiment-plan.json`指定。C3不把server容量改成3。
- `kv6p75.json`：6.75GiB；`chunk2048.json`：KV6.5GiB＋chunk2048；`memory.json`：独立MemoryAuditWorker，6.25GiB。
- `experiment-plan.json`：历史顺序、准入9000输出预留、三轮完整重复、所有尝试计入有效吞吐、至少2%才换默认的采用规则。
- `kv-scales.json`：固定数值scales，已移除私有校准来源；`model-fingerprint.json`：沿用的模型文件指纹。

模型/别名/端口为可移植占位值；start.py设置实际模型路径和端口。显式KV字节预算覆盖显存比例；扩大前核算本机空闲与Graph池，不照抄其他机器的安全余量。


[返回方案](../readme.md)
