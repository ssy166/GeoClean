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
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-60}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

mkdir -p logs results/smoke

attempt=1
until timeout 20m .venv/bin/python scripts/download_models.py --models flux --resume; do
  status=$?
  echo "[$(date --iso-8601=seconds)] FLUX download attempt ${attempt} exited with ${status}; retrying in 10s"
  attempt=$((attempt + 1))
  sleep 10
done

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

echo "[$(date --iso-8601=seconds)] FLUX smoke test complete"
