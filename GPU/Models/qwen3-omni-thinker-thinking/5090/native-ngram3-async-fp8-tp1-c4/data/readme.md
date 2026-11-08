# 自备且冻结性能输入

公开包不提供业务媒体、system prompt、shot plan 原文、Caption 或原数据来源。manifest.example.json 只是结构示例，SHA 占位符必须换成真实值，不能直接运行复现实验。

```text
$DATA_DIR/
  sample-001/
    video.mp4
    img01.png
    img02.png
    img03.png
    ...                     # 4–5 张时全部使用
$PREPARED_DIR/
  manifest.json             # 列表，参照 manifest.example.json
  sample-001/
    system-prompt.txt        # 最终完整 system 字符串，含已冻结 runtime context
    runtime-context.json    # 冻结镜头表与运行时 JSON，供核查
```

客户端从 `$DATA_DIR/<sample_id>/video.mp4` 和 reference_images.file 读媒体，从 prepared_subdir/system-prompt.txt 读实际 system；检查每个媒体 SHA、最终 system SHA。不要在对照中重新切镜头或改 prompt。原测试 shot plan 自动生成，未人工确认切分准确性。

原性能集 15 个视频，12.04–15.04 秒，总 207.2633 秒；共 64 张参考图，每条全部 3–5 张；原同步音轨，2 fps 采样，完整长 system prompt，15 条每轮输入 419,460 tokens。每轮取最大输入 37,321 tokens 的一条完整预热，再 C4 测全部 15 条，两轮/条件。

新数据可以按合同运行，但结果单独记录；没有原冻结业务输入访问权就不能严格复现相同数值。性能集无人工语义标签，不能用 JSON 完整率冒充 Caption 精度。

[输入与计时协议](../readme.md) · [匿名指标](../evidence/readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
