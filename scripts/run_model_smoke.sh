#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -f .env ]]; then
  set -a
  source .env
  set +a
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET="${HF_HUB_DISABLE_XET:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs results/smoke

echo "[$(date --iso-8601=seconds)] downloading model snapshots"
.venv/bin/python scripts/download_models.py --models flux sd35 --resume

echo "[$(date --iso-8601=seconds)] running FLUX smoke sample"
.venv/bin/python sample/baseline_flux.py \
  --model_path models/FLUX.1-dev \
  --csv_path data/i2p_benchmark.csv \
  --methods base \
  --num_samples 1 \
  --height 512 \
  --width 512 \
  --num_inference_steps 4 \
  --seed 42 \
  --device cuda \
  --output_dir results/smoke/flux

echo "[$(date --iso-8601=seconds)] running SD3.5 Medium smoke sample"
.venv/bin/python sample/sd35_medium_sample.py \
  --model_path models/stable-diffusion-3.5-medium \
  --height 512 \
  --width 512 \
  --num_inference_steps 4 \
  --seed 42 \
  --device cuda \
  --output results/smoke/sd35_medium.png

echo "[$(date --iso-8601=seconds)] smoke tests complete"
