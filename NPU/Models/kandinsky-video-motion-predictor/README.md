# Kandinsky Video Motion Predictor · Ascend 910B1

单张910B1 64GB。原FP32权重、FP16 autocast，物理MP4/torch预处理；默认8×2/FIFO，优化可配置关闭。
同100条精简代码ABBA：47.246→12.650秒，耗时降低73.22%，吞吐7.905条/秒；
各轮像素哈希与原NPU分数精确一致。跨GPU仍有原有FP16数值差异，整体MSE5.91317e-6。

[完整交接](HANDOFF.md) · [环境](environment/README.md) · [输入合同](INPUTS.md) ·
[性能/精度/回退报告](reports/RESULTS.md) · [GPU方案](../../../GPU/Models/kandinsky-video-motion-predictor/README.md)

## 启动

以下变量由部署者设置，先检查目标卡空闲；不要覆盖共享驱动栈或占用其他业务卡。

```bash
python scripts/fetch_model_source.py --destination "$UPSTREAM_DIR"
export MOTION_UPSTREAM_DIR="$UPSTREAM_DIR" QC_CKPT_DIR="$CHECKPOINT_ROOT"
export ASCEND_RT_VISIBLE_DEVICES="$NPU_ID" OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
python scripts/verify_weights.py --model-dir "$CHECKPOINT_ROOT/ai-forever/kandinsky-video-motion-predictor"
python runtime/run_motion.py --code runtime/code --manifest "$MANIFEST" \
  --weights-root "$CHECKPOINT_ROOT" --config configs/accepted_config.json --warmup --output "$NEW_OUTPUT"
```

| 配置 | CPU workers | reader解码线程 | 提交顺序 | 复用时长 |
|---|---:|---:|---|---|
| 不传 / baseline_config.json | 8 | 2 | FIFO | 关闭 |
| accepted_config.json | 16 | 4 | 离线LPT | 开启 |
| online_config.json | 16 | 4 | FIFO | 开启 |

LPT需width/height/num_frames，按像素帧数估计成本，大任务先入池；结果恢复原清单顺序。
在线未来请求不能预知，使用FIFO配置；工具不提供HTTP服务，不把批吞吐当在线SLA。
时长优先motion_duration，其次duration_s；缺失退回原PyAV元数据读取，错误时长可能改变抽帧。

模型和CPU池各初始化一次。`--warmup`预热代表帧数，模型加载、池就绪、预热单列；
`request_wall_s`包含MP4处理、进程传输、推理和可选精确校验，不含初始化、结果文件写出和退出。
追加`--verify-pixels "$REFERENCE_JSONL" --verify-scores "$NPU_REFERENCE_ARRAY_JSON"`执行逐值检查，
失败直接报错，不自动重试、填零或恢复中间结果。

换`baseline_config.json`或不传配置回原路径。原地归一化、Head快路径、常驻AMP未达2%收益门槛，
实现已移除，尝试启用其配置会被拒绝。默认prep直接委托原函数，不改模型数学路径。

16×4比8×2增加CPU/RAM需求；每worker另设torch4线程，参数乘积不是实测物理核利用率。
新鲜进程离线入口PSS峰值约34.05GiB，PyTorch NPU allocated峰值1.82GiB、reserved2.97GiB。
常驻池allocator保留可能更高；短批不证明长期无泄漏。所有实测worker结束后退出，NPU回驱动基线。
