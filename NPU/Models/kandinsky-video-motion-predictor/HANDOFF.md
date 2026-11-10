# Motion Predictor 交接

## 范围与状态

单卡物理MP4的Kandinsky VideoMotionPredictor，输出camera/object/dynamics三项运动强度分数。
GPU保留历史单实例/逐clip Head及CPU筛选协议；NPU增加可关闭的CPU并发、解码线程、
离线LPT和可信容器时长复用。模型、权重和数值路径不变，失败直接退出，不自动重试。
原生产虚拟切片、CFR映射、cv2 fast融合、数据库写入和网络在线服务不属于本交接实现。

## 目录

| 位置 | 内容 |
|---|---|
| runtime/code/motion_only.py | 原物理MP4准备和GPU模型调用；仅改为显式权重/上游路径 |
| runtime/run_motion.py（NPU） | 常驻模型+有界CPU池批处理入口 |
| runtime/motion_pipeline.py（NPU） | 可选调度/预处理配置，默认委托原路径 |
| runtime/baseline_npu_motion.py（NPU） | 原单clip Head语义的NPU adapter |
| scripts/benchmark.py（GPU） | 实测CPU筛选及100条协议；只改命令行路径接入 |
| scripts/make_reference.py（GPU） | 用自有视频生成本地串行参考及固定输入fixture |
| scripts/compare_scores.py | 无需模型的MSE/RMSE/MAE/P95/偏差/相关性/R²与阈值比较 |
| scripts/fetch_model_source.py | 按固定公开revision获取8个模型文件，SHA256校验 |
| scripts/verify_weights.py | 校验已有checkpoint，不下载或分发权重 |
| configs/ | NPU可执行配置、GPU已测配置描述 |
| reports/ | 聚合性能、精度、内存与优化决策，不含逐样本输出 |
| tests/test_compare.py | 无硬件统计及配对合同测试 |

实际目录以所在设备README为准；GPU/NPU部分入口互不混用。
环境、上游固定版本和权重校验见[environment/README.md](environment/README.md)，
输入规则见[INPUTS.md](INPUTS.md)，结果与计时边界见[reports/RESULTS.md](reports/RESULTS.md)。

## 执行顺序

1. 创建独立环境，安装与设备匹配的驱动/torch栈及通用依赖，不覆盖共享环境。
2. 获取固定上游8文件、校验权重；设置MOTION_UPSTREAM_DIR、QC_CKPT_DIR。
3. 核对输入manifest及原GPU/NPU参考；先检查目标卡和CPU/RAM余量。
4. 按所在目录README启动。模型和池初始化单列，预热后计时；每次输出用新目录。
5. 核对成功数、像素哈希和分数，不把wrapper exit0等同于全量通过。
6. 退出后检查worker/NPU计算进程和驱动基线；长服务额外观察同形状排空后的allocated/PSS。

比较自有GPU/NPU或优化前后结果：

```bash
python -m unittest discover -s tests -p 'test_*.py'
python scripts/compare_scores.py --reference "$GPU_SCORES" --candidate "$NPU_SCORES" \\
  --require-input-hashes --output "$NEW_COMPARISON"
```

仅对要求零新增误差的NPU优化前后比较增加`--require-exact-scores`。
原跨硬件FP16存在误差，不应给该比较加零误差开关。工具报告聚合指标，不擅自设置新的精度验收门槛。
参考可为JSON数组或JSONL，须严格同一clip_id集合；不存在真值、不能把GPU当人类标签。

## 回退与边界

NPU用baseline_config.json或不传config返回8×2/FIFO/原元数据读取；原地归一化、
Head快路径、常驻AMP均已从实现移除。CPU16×4增加资源，LPT只用于已知离线批次；在线接入用FIFO候选。
GPU原路径不新增NPU优化项，不把配置描述JSON当可执行benchmark参数。
没有更改权重、量化或attention；没有引入共享内存、DVPP、自定义resize和新的调度服务。

发布时做源码函数/AST对照：原MotionScorer.prep和infer_batch保持一致；
NPU三个已测runtime文件保持AST一致；公开模型8文件匹配固定上游提交。
新增工具做CLI/CPU测试和私有已存分数回放，不宣称发布时重跑所有历史GPU筛选或在全新环境从零部署。
各次实际运行和源码对照范围分别记录，不能将静态检查当硬件精度验证。

## 精度与样本限制

100条中输入像素哈希全相同；最优NPU vs 原NPU MSE=0，GPU并发 vs GPU串行MSE=0。
最优NPU vs GPU整体MSE=5.91317e-6、RMSE=0.00243170、最大绝对误差0.0131836。
已有static/violent规则均零翻转，但仅2条static阳性、没有violent阳性，未覆盖全部边界。
这是数值一致性评估，不是人工运动预测准确率；其他编解码、长视频、分布和硬件须重新校准。

私有视频、npy、逐条分数、真实sample ID、原生trace和服务器日志留在操作者本地。
公开聚合证据不能重建该100条私有cohort；使用自有数据可复现执行流程，不保证复现相同耗时。
