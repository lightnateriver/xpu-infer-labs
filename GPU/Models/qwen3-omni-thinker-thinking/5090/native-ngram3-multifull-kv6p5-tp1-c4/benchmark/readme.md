# 请求、预热和准入客户端

`bench_caption.py`、`bench_fixed.py`、`native_adapter.py`、`admission.py`与实测源码保持一致；`_common.py`复用已有公开的receipt-only原生适配，不携带主机特定Docker路由或凭证提取。所有访问使用调用方环境中的VLLM_API_KEY。

自然客户端使用全部参考图、原视频音轨、冻结prompt，流式返回Thinking/正文，max_tokens=24576，temperature=0，seed=0。无效自然轮先完成API计数审核再报错，包装脚本保留其有效数与wall并补测。

固定客户端选择manifest下标0/4/9/14、ignore_eos、每条4096输出，只有预热/效率/Graph检查用途，不能纳入自然Caption质量。此测试套件要求15条冻结输入。

TokenGate按manifest先后FIFO：实际KV容量乘比例、每条预留prompt+9000token。仅正式自然轮启用。9000是经验预留，不限制生成或保证零抢占。排队计入提交到结束E2E和批次wall。

底层入口参数为`--env-file /dev/null --run-dir <private-run> --data-dir <private-data> --container <receipt-label> --endpoint <loopback-endpoint> --alias <alias> --gpus <physical-index> --concurrency 3或4 --name <fresh-label> --phase preflight/warmup/measured`。运行前设置BENCH_NATIVE_RECEIPT、BENCH_KV_CAPACITY、BENCH_ADMISSION_RATIO、BENCH_OUTPUT_RESERVE；具体路径设置和依赖文件布局见[scripts/benchmark.py](../scripts/benchmark.py)。每个phase名必须新建，禁止同端口其他生成请求。这里的container仅为适配标签，服务为原生进程。


[返回方案](../readme.md)
