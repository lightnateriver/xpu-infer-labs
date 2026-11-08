# 2026-11-08 设备目录归档

迁移前 main：`32e6fd2ba914fadc17dd7d2fc399ebd3cde1a77a`。
归档标签：`archive/pre-device-layout-20261108`。保留 Git 历史，仓库继续可写。

| 原目录 | 新目录 |
|---|---|
| `models/qwen3-omni-thinker-thinking` | `GPU/Models/qwen3-omni-thinker-thinking` |
| `models/vllm-omni-h3` | `GPU/Models/minimax-h3` |
| `models/qwen3-omni-instrcut/npu` | `NPU/Models/qwen3-omni-instruct` |

修正 Instruct 拼写，MiniMax-H3 按模型名归档。`models/readme.md` 保留迁移指引。
新增 Qwen3-VL-Embedding-2B 的 GPU / NPU 独立交接目录。旧模型的版本、质量标签、报告和第三方许可证保留；目录移动不改变其历史验收范围。
