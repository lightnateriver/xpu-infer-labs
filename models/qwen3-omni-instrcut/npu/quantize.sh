#!/usr/bin/env bash
set -euo pipefail
source_model=${1:?Pass the official BF16 checkpoint directory}
output=${2:?Pass a new output directory}
test ! -e "$output"
export ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
msmodelslim quant --model_path "$source_model" --save_path "$output" --device npu \
  --model_type Qwen3-Omni-30B-A3B-Instruct --quant_type w8a8 --trust_remote_code true
for file in "$source_model"/*.{json,txt,model,jinja}; do
  [[ -f "$file" && "$file" != *safetensors* ]] || continue
  cp -n "$file" "$output/"
done
test -s "$output/quant_model_description.json"
compgen -G "$output/*.safetensors" >/dev/null
