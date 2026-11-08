# 独立性能报告

[report.html](report.html)为无外部依赖的红色浅色单页，含7方案有效吞吐柱状图、TTFT/TPOT/E2E、逐轮失败项、每卡TPS、显存核算及7组KV时间线。Thinking计入时间；方法、数据集与精度边界完整说明。

从匿名证据重建：

```bash
python3 report/build_report.py
```

生成脚本只读取本方案evidence，不读取私有文件或其他HTML。报告数值来自已完成GPU实验；本次发布新增包装脚本未重新运行GPU。浏览器视觉检查未执行，公开报告已进行结构、依赖、数据与隐私审阅。


[返回方案](../readme.md)
