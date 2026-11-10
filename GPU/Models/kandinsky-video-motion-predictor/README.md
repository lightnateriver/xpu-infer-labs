# Kandinsky Video Motion Predictor · RTX 5090

单卡物理MP4运动幅度推理；原FP32权重、FP16 autocast，camera/object/dynamics三项输出。
已测最优CPU配置为4个预处理进程、每reader2解码线程、worker不绑核；模型单实例、逐clip Head。
同100条视频53.200秒，1.880条/秒。NPU后续针对解码和离线调度另做优化，不能把整机比例当芯片比例。

[完整交接](HANDOFF.md) · [环境](environment/README.md) · [输入合同](INPUTS.md) ·
[性能/精度/回退报告](reports/RESULTS.md) · [NPU方案](../../../NPU/Models/kandinsky-video-motion-predictor/README.md)

## 启动

以下变量由部署者设置；权重和输入自备，先检查卡空闲。每个output目录必须不存在。

```bash
python scripts/fetch_model_source.py --destination "$UPSTREAM_DIR"
export MOTION_UPSTREAM_DIR="$UPSTREAM_DIR" QC_CKPT_DIR="$CHECKPOINT_ROOT"
export CUDA_VISIBLE_DEVICES="$GPU_ID" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python scripts/verify_weights.py --model-dir "$CHECKPOINT_ROOT/ai-forever/kandinsky-video-motion-predictor"
python runtime/code/run_single.py --video "$VIDEO" --output "$SINGLE_OUTPUT_JSON"
```

历史CPU筛选协议需要100条/10来源的自有清单及本机串行参考。先生成参考（其耗时不属于批吞吐），再运行：

```bash
python scripts/make_reference.py --code runtime/code --manifest "$DATASET/clips.jsonl" --output "$REFERENCE_DIR"
python scripts/benchmark.py --code runtime/code --fixture-dir "$REFERENCE_DIR/validation" \
  --dataset "$DATASET" --baseline "$REFERENCE_DIR" --output "$NEW_OUTPUT"
```

清单clip_id按`source/clip`组织，索引唯一；历史筛选每来源最短/最长共20条，依次试
1/free、2/pinned、4/free、4/pinned、8/pinned，再选择最快配置处理100条。
父进程从低负载物理核中选4核；worker不绑核配置使用父进程选择前的可用CPU集合。
需至少36个可用物理核，Linux `/proc` 和lscpu；不适用于Windows直接推理。
CPU池和模型常驻；最多2×workers在途任务，CPU worker隐藏CUDA。

这里保留实测GPU执行函数；仅将原私有目录参数拆成`--code`、`--fixture-dir`及显式环境变量。
公开源码函数对照、CLI/CPU检查见HANDOFF；不声称发布时重跑了原100条GPU筛选。
单视频入口不等于常驻批吞吐，GPU尚未用NPU的16×4/LPT策略重测。
