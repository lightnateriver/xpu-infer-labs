# 独立性能报告

打开本目录 report.html 即可查看；没有外部 JS/CSS、网络字体、关联 HTML 或私有媒体。浅色红系，包含吞吐柱状图、每卡 TPS、TTFT/TPOT/E2E、逐轮与 90 条匿名指标、接受率、输入定义、配置变化及质量边界。

```bash
python3 report/build_report.py
```

从 ../evidence/results.json 重建；生成不访问 GPU，也不是新一轮性能测试。N3 的 2.27% 任务吞吐增益是两轮记录，未证明稳定生产收益；N5 不作为推荐配置。

[返回方案说明](../readme.md)

## 后续多 FULL Graph / KV 更新（2026-10-21）

[当前方案](../../native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、Thinking计入、3.575有效条/分钟，完整轮310.22TPS/卡中位数、45/45完整、零抢占；相对其本轮C4基线+10.33%。本目录的历史结果和代码保留，新实验输出上限/测量协议不同，不能跨报告直接归因，也未证明Caption语义精度无损。
