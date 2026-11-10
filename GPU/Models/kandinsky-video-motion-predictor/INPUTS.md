# 输入、输出和私有参考合同

输入为物理 MP4，每行一个 JSON 的 UTF-8 JSONL；使用自有素材和匿名 clip_id。
示意字段（路径由调用方传入，示例并非随库提供的数据）：

```json
{"clip_id":"clip-0001","path":"videos/clip-0001.mp4","duration_s":5.0,"width":1280,"height":720,"num_frames":125,"fps":25.0}
```

`path` 相对于启动进程工作目录，或由调用方传入本机路径；仓库没有硬编码部署路径。
`duration_s` 必须为当前物理切片的容器时长，不可误用原始长视频时长。
LPT 需要正数 width/height/num_frames；FIFO 不依赖这些调度字段。

抽帧规则保持原样：`num=int((min(4*int(duration),240)//16+1)*16)`，
`linspace(0,total_frames-1,num).long()`，保留重复帧。上限实际为256帧，最少16帧。
不保持宽高比，torch bilinear 缩至224×224；量化uint8后进程传输，模型端float32/255。
Backbone可处理多个16帧segment，但Transformer head按clip单独运行，避免padding改变短片得分。

私有参考JSONL每行：`clip_id`、`input_sha256`、`scores`；scores有
motion_camera、motion_object、motion_dynamics三个有限数。像素SHA256针对连续
uint8 `[T,3,224,224]` 的原始字节。NPU的`--verify-scores`接受同字段的JSON数组；
`--verify-pixels`接受JSONL。GPU历史benchmark的`--baseline`目录需含cases.jsonl。

输出包含分数、耗时、输入哈希和进程内存快照，可能关联私人数据，应保存在未跟踪的本地目录。
公开报告仅保留聚合指标，不包含真实清单、视频、输入张量、逐条分数或样本标识。
没有独立人工运动强度真值；与GPU数值接近并不等于任务预测正确率。
原生产虚拟切片/cv2 fast路径与这里的物理MP4/torch路径不同，未集成本轮优化。
