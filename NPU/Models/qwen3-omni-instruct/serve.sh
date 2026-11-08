#!/bin/bash
set -e
ulimit -c 0
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 OMP_PROC_BIND=false HCCL_BUFFSIZE=512 HCCL_OP_EXPANSION_MODE=AIV
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 VLLM_NO_USAGE_STATS=1
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/tmp/vllm-cache}
exec vllm serve "${MODEL_DIR:-/model}" --host 127.0.0.1 --port "${PORT:-18096}"  --served-model-name qwen3-omni-instruct --tensor-parallel-size 1 --quantization ascend --dtype bfloat16 --distributed-executor-backend mp  --max-model-len 65536 --max-num-seqs 8 --max-num-batched-tokens 4096  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 21474836480  --mm-processor-cache-gb 0 --generation-config vllm --no-enable-prefix-caching --async-scheduling  --limit-mm-per-prompt '{"image":5,"video":1,"audio":1}'  --skip-mm-profiling  --compilation-config '{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,4,8]}'  --additional-config '{"enable_flashcomm1":false}'
