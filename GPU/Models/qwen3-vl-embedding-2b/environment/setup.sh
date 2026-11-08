#!/usr/bin/env bash
set -euo pipefail
# Use a Python 3.12 virtual environment with a compatible CUDA driver.
python -m pip install 'torch==2.10.0' 'torchvision==0.25.0' 'vllm==0.19.0' \
  'transformers==4.57.6' 'qwen-vl-utils==0.0.14' 'av==17.1.0' \
  'numpy==2.2.6' 'pillow==13.0.0' 'imageio-ffmpeg==0.6.0'
