# xpu-infer-labs

GPU / NPU 推理优化实验、可运行交接代码与匿名验证证据。模型配方按设备归档：

| 设备 | 模型 | 入口 |
|---|---|---|
| GPU · 单卡 RTX 5090 | Qwen3-VL-Embedding-2B | [CPU32 视频嵌入流水线](GPU/Models/qwen3-vl-embedding-2b/README.md) |
| NPU · 单卡 Ascend 910B1 | Qwen3-VL-Embedding-2B | [稳定单遍版 / NPU-SPG v1](NPU/Models/qwen3-vl-embedding-2b/README.md) |
| GPU · RTX 5090 / GB300 | Qwen3-Omni Thinking | [模型索引](GPU/Models/qwen3-omni-thinker-thinking/readme.md) |
| GPU · RTX 5090 | MiniMax-H3 | [V1 SP8 / SP4 交接](GPU/Models/minimax-h3/5090/readme.md) |
| NPU · Ascend 910B3 | Qwen3-Omni Instruct | [单卡 W8A8](NPU/Models/qwen3-omni-instruct/readme.md) |

[GPU 索引](GPU/Models/README.md) · [NPU 索引](NPU/Models/README.md) · [目录迁移记录](ARCHIVE.md)

Qwen 视频嵌入交接包含 runtime、配置、环境来源、普通文件/TAR 清单测试、CPU 防错测试与性能/精度报告。
**同输入余弦 ≥0.99 与旧向量库在 0.99 阈值处的判定一致性是两个门槛。** NPU-SPG v1 是前者下的最快候选，不能据此声称旧库零跨线。

其他模型保留原始历史证据和测试范围，本次目录迁移不代表重新测试了全部模型。
不发布私有素材、样本标识、原始业务输出、服务器地址或访问凭证。复现者需自备模型和有权使用的冻结输入；匿名汇总不足以重建私有测试集。

## Qwen 视频嵌入本次重测

GPU CPU32 7.065 条/秒，NPU-SPG v1 8.110 条/秒，后者为前者的 **114.79%**。同输入余弦 ≥0.99 通过，旧库 0.99 阈值零跨线未通过；跨架构像素也非字节一致。[完整报告与限制](NPU/Models/qwen3-vl-embedding-2b/reports/retest-20261108.md)。
