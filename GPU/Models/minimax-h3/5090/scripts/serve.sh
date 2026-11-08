#!/usr/bin/env bash
set -euo pipefail
root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
profile=${1:-}
: "${MODEL_DIR:?Set MODEL_DIR to the local MiniMax-H3 checkpoint directory}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_OMNI_VIDEO_SYNC_TIMEOUT=3600
unset H3_V2_MODE_FILE H3_V2_ADALN_OWNER H3_ONLINE_FP8 H3_V2_NVFP4_ATTN H3_V2_CHECK_ADALN_TIMESTEPS H3_V3_KEEP_REFINER

args=("$MODEL_DIR" --omni --task-type fl2va --trust-remote-code
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8091}"
    --diffusion-attention-backend CUDNN_ATTN --vae-parallel-mode tile --vae-use-tiling)
case "$profile" in
    v1-sp8|sp4-alternate)
        : "${H3_OVERLAY:?Set H3_OVERLAY to the output of prepare_overlay.py}"
        H3_OVERLAY=$(cd -- "$H3_OVERLAY" && pwd)
        export PYTHONPATH="$H3_OVERLAY"
        export H3_V2_MODE_FILE="$H3_OVERLAY/mode.txt" H3_V2_ADALN_OWNER=1 H3_ONLINE_FP8=1
        cmp "$H3_V2_MODE_FILE" "$root/configs/$profile.mode"
        python - "$root/patches/patched-sha256.json" "$profile" "$H3_OVERLAY" <<'PY'
import hashlib
import json
from pathlib import Path
import sys

import vllm_omni
from flashinfer.diffusion_ops import minimax_h3_sm120_varlen_attention_nvfp4

manifest, profile, overlay = sys.argv[1:]
overlay = Path(overlay)
assert Path(vllm_omni.__file__).resolve().is_relative_to(overlay)
for name, expected in json.loads(Path(manifest).read_text(encoding="utf-8-sig"))[profile].items():
    if hashlib.sha256((overlay / name).read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Wrong overlay source: {name}")
PY
        args+=(--tensor-parallel-size 1 --ring 1 --ulysses-mode strict
            --enable-distributed-layerwise-offload --dlo-no-use-allgather
            --dlo-resident-layers 50 --init-timeout 1800 --stage-init-timeout 1800)
        if [[ "$profile" == v1-sp8 ]]; then
            args+=(--num-gpus 8 --usp 8 --text-encoder-tp-size 8 --vae-patch-parallel-size 8
                --quantization fp8)
        else
            : "${CUDA_HOME:?Set CUDA_HOME to the CUDA toolkit used to build FlashInfer kernels}"
            export PATH="$CUDA_HOME/bin:$PATH"
            args+=(--num-gpus 4 --usp 4 --text-encoder-tp-size 4 --vae-patch-parallel-size 4
                --linear-backend cutlass --diffusion-quantization-config "$(cat "$root/configs/sp4-quantization.json")"
                --cache-config '{"minimax_h3_adaln_cache":false}')
        fi
        ;;
    native-bf16-tp8)
        : "${OMNI_SOURCE:?Set OMNI_SOURCE to the clean vLLM-Omni checkout}"
        test -f "$OMNI_SOURCE/vllm_omni/__init__.py"
        export PYTHONPATH="$(cd -- "$OMNI_SOURCE" && pwd)"
        export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
        args+=(--num-gpus 8 --tensor-parallel-size 8 --text-encoder-tp-size 8
            --vae-patch-parallel-size 8 --dtype bfloat16 --enforce-eager)
        ;;
    *)
        printf 'Usage: %s {v1-sp8|sp4-alternate|native-bf16-tp8}\n' "$0" >&2
        exit 2
        ;;
esac
exec vllm serve "${args[@]}"
