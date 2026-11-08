# 冻结输入与精度参考

清单格式如下。普通视频文件使用 offset=0、size=整个文件字节数；TAR 成员使用真实 payload 偏移与长度。
SHA256 必须是该 payload 的哈希。路径相对清单目录或为绝对路径。所有 warmup/cohort ID 不重复，各组顺序冻结。

```json
{
  "warmup": [{"sample_id":"warm-000", "path":"warm.mp4", "offset":0, "size":1234,
    "sha256":"REPLACE_WITH_REAL_SHA256", "height":720, "width":1280, "num_frames":240, "fps":24.0}],
  "cohorts": {
    "screen": [{"sample_id":"screen-000", "path":"videos.tar", "offset":512, "size":2345,
      "sha256":"REPLACE_WITH_REAL_SHA256", "height":720, "width":1280, "num_frames":240, "fps":24.0}],
    "holdout": [{"sample_id":"holdout-000", "path":"other.mp4", "offset":0, "size":3456,
      "sha256":"REPLACE_WITH_REAL_SHA256", "height":720, "width":1280, "num_frames":240, "fps":24.0}]
  }
}
```

以上仅格式示例，数字不是真实素材元数据。正式对比建议各组 64 条、独立 warmup 16 条；每个 worker 解码一条真实 warmup 并经 barrier 后，整体预热，再对每组完整预热一次，随后计时三轮。
这些协议与一次调用 128 并发不同。输入必须完整可读，metadata 与解码帧数不符即拒绝。

可用 `--references /private/reference-map.json` 指定精度对照：

```json
{"screen":{"query":"/private/screen-reference.npy","gallery":"/private/screen-gallery.npy",
  "stable":"/private/screen-stable-npu.npy","contract":"/private/screen-contract.json"},
 "holdout":{"query":"/private/holdout-reference.npy","gallery":"/private/holdout-gallery.npy"}}
```

每个 `.npy` 为 N×2048 float32，行顺序严格对应输入。`query` 是冻结 GPU query，`gallery` 是冻结旧 GPU 库；`stable` 和 `contract` 可选。
脚本报告同输入余弦、组内非自身有向阈值变化、相对稳定 NPU 的字节差异。合同检查忽略 ID 文本，但核对帧索引、形状、prompt token SHA256。
完整两组合并的精度统计还应拼接 query/gallery 后重新计算，不能只把组内跨线数相加；本次报告采用完整 128×128 非自身矩阵。
不要重新生成冻结参考来消除精度差异。向量、视频、逐样本元数据和路径均属于本地私有输出。

使用 `scripts/compare.py` 可在合并两组后重算完整矩阵，传入 `--runs label=/results/run`、`--reference-map /private/reference-map.json` 与未存在的 `--output`。每个 run 的 cohort 顺序和参考必须相同。该工具同时报告帧/形状/token 合同和预处理组合哈希，跨架构哈希可能不完全一致，参见重测报告。
