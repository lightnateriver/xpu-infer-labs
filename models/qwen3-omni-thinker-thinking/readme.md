# Qwen3-Omni Thinker / Thinking

[GB300 方案](GB300/readme.md)：官方 Thinking 源权重，单卡 TP1/C56，NVFP4 experts、FP8 KV128GiB、FULL_DECODE_ONLY、async off。历史 56/56 完整，**15.358 条/分钟，折合 3.907 秒/条吞吐**；单请求中位耗时 **127.39 秒**，含 Thinking。目录提供部署/测试脚本、固定 runtime 和匿名证据，发布时未重新测速；不与下述 RTX5090 不同权重/协议直接排名。

[RTX5090索引](5090/readme.md)列出当前与历史方案。

当前采用[原生N3/K3＋多FULL Graph＋KV6.5GiB](5090/native-ngram3-multifull-kv6p5-tp1-c4/readme.md)：TP1/C4、3.575有效条/分钟、完整轮310.22TPS/卡中位数，Thinking计入，45/45完整、零抢占。相对本轮C4基线+10.33%。使用非官方Thinking-heretic-NVFP4，attention线性层FP8、MoE NVFP4、FP8 KV；不是官方BF16全模型。

保留历史[原生N3](5090/native-ngram3-async-fp8-tp1-c4/readme.md)与[Omni FP8](5090/async-fp8-gemm-fp8-kv-tp1-c4/readme.md)。新实验输出上限24576、旧N3为16384，跨报告数值不能直接归因。完整性和Graph数值校验不等于人工Caption语义精度。
