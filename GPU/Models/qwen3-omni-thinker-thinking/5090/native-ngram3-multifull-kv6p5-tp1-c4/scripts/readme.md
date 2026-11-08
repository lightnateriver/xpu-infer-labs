# 操作脚本

- `start.sh`/`start.py`：默认kv6p5；`--variant c4/c3/budget90/budget95/kv6p5/kv6p75/chunk2048/memory`。配置计算参数来自configs，模型/端口/选卡由调用方指定。核对模型、空卡、版本/源码、runtime审计和26张捕获。
- `stop.py`：仅处理receipt记录的UID/PID/start ticks/进程会话；不处理未知作业。
- `benchmark.sh`/`benchmark.py`：新私有run-dir，预检、4条固定输出预热/Graph检查，默认至少3轮完整自然结果，最多9次，包含全部正式尝试wall。memory变体走一次15条独立诊断。
- `memory_capacity.py`：空闲且归属匹配后发出诊断指令，按实际峰值/Graph池/非Torch与512MiB余量给出KV候选资格。
- `summarize.py`：汇总所有自然尝试；有效条/分钟与完整轮median TPS分开。
- `check_environment.py`：只读核对版本、关键源码和工具，不创建CUDA上下文。

这些可移植包装脚本在发布时新增/改造，未再次进行GPU运行测试；实测的Worker和请求客户端身份另有记录。历史交错方案的完整时间顺序见configs计划；当前包装脚本按每方案连续重复，不冒充逐时复刻。所有本地状态/日志/输入输出写入忽略的.local目录。具体命令见[主README](../readme.md)。


[返回方案](../readme.md)
