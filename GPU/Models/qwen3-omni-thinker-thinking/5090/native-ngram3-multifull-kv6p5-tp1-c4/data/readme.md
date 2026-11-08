# 输入合同

自备15条有使用权限的冻结视频，每条目录包含video.mp4和全部参考图；prepared目录包含manifest.json及每条prepared_subdir/system-prompt.txt。完整system prompt已融合冻结ShotPlan/runtime context，不在客户端重新切分。`manifest.example.json`仅为字段示例，SHA占位必须替换真实值，不能直接作为完整测试集。

历史性能集15视频、64参考图（3–5张/条）、2fps、原同步音轨，共419460输入tokens/轮；实际视频约12–15秒。manifest应按原输入顺序冻结，保存视频/图片/system SHA、prepared_subdir、duration_seconds、shot_count等。客户端核对内容和token数。内容、原名称/来源和业务prompt不公开，公开指标将样本统一为sample-001至sample-015；新数据的性能另列，不能声称复现历史数值。

这是性能集，不是带人工真值的Caption精度集。完整性标准只检查流、Thinking、输入/输出预算、JSON及约定结构/ID等，不判断镜头、人物、时间或声音描述是否真实。


[返回方案](../readme.md)
