#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/stable-diffusion-3.5-medium}"
GPU_IDS="${GPU_IDS:-5}"
GPU_MEM_LIMIT_MB="${GPU_MEM_LIMIT_MB:-4096}"
CHECK_INTERVAL_SEC="${CHECK_INTERVAL_SEC:-300}"
SD35_NUM_SAMPLES="${SD35_NUM_SAMPLES:-50}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
STEPS="${STEPS:-28}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-4.5}"
SAFETY_SCALE="${SAFETY_SCALE:-1.5}"
ACS_DELTA="${ACS_DELTA:-0.2}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"

LOG_DIR="$ROOT/logs/rebuttal_sd35"
OUT_DIR="$ROOT/results/rebuttal_sd35"
mkdir -p "$LOG_DIR" "$OUT_DIR"

log() {
  echo "[$(date -Is)] $*"
}

sd35_download_running() {
  pgrep -f "scripts/download_models.py --models sd35" >/dev/null 2>&1
}

required_sd35_files_present() {
  [[ -f "$MODEL_PATH/model_index.json" ]] &&
  [[ -f "$MODEL_PATH/transformer/config.json" ]] &&
  [[ -f "$MODEL_PATH/transformer/diffusion_pytorch_model.safetensors" ]] &&
  [[ -f "$MODEL_PATH/vae/config.json" ]] &&
  [[ -f "$MODEL_PATH/vae/diffusion_pytorch_model.safetensors" ]] &&
  [[ -f "$MODEL_PATH/tokenizer/tokenizer_config.json" ]] &&
  [[ -f "$MODEL_PATH/tokenizer_2/tokenizer_config.json" ]] &&
  [[ -f "$MODEL_PATH/tokenizer_3/tokenizer_config.json" ]]
}

wait_for_download() {
  while true; do
    if required_sd35_files_present && ! sd35_download_running; then
      log "SD3.5M files are present and downloader is no longer running."
      return
    fi
    local size
    size="$(du -sh "$MODEL_PATH" 2>/dev/null | awk '{print $1}' || true)"
    if sd35_download_running; then
      log "Waiting for SD3.5M download to finish. Current size: ${size:-unknown}."
    else
      log "Downloader is not running yet required files are incomplete. Current size: ${size:-unknown}."
    fi
    sleep "$CHECK_INTERVAL_SEC"
  done
}

pick_free_gpu() {
  local gpu used
  for gpu in $GPU_IDS; do
    used="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" | tr -d ' ')"
    if [[ -n "$used" && "$used" -lt "$GPU_MEM_LIMIT_MB" ]]; then
      echo "$gpu"
      return 0
    fi
  done
  return 1
}

wait_for_gpu() {
  local gpu=""
  while true; do
    if gpu="$(pick_free_gpu)"; then
      echo "$gpu"
      return
    fi
    log "Waiting for a free GPU among: $GPU_IDS."
    nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits
    sleep "$CHECK_INTERVAL_SEC"
  done
}

run_sd35() {
  local gpu="$1"
  export CUDA_VISIBLE_DEVICES="$gpu"
  log "Using physical GPU $gpu for SD3.5M validation."

  log "Running SD3.5M smoke test."
  "$PYTHON_BIN" sample/sd35_medium_sample.py \
    --model_path "$MODEL_PATH" \
    --output "$OUT_DIR/smoke/sd35_medium_smoke.png" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_inference_steps 4 \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --device cuda

  log "Running SD3.5M I2P baseline/SLD/GeoClean validation."
  "$PYTHON_BIN" sample/baseline_sd35.py \
    --model_path "$MODEL_PATH" \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --output_dir "$OUT_DIR/i2p_s15_full" \
    --methods base sld geoclean \
    --num_samples "$SD35_NUM_SAMPLES" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --concept nudity \
    --safety_scale "$SAFETY_SCALE" \
    --sld_threshold 0.0 \
    --acs_delta "$ACS_DELTA" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --device cuda

  log "SD3.5M validation finished."
  "$PYTHON_BIN" scripts/summarize_rebuttal_runs.py --root "$OUT_DIR" || true
}

log "SD3.5M watcher started."
log "MODEL_PATH=$MODEL_PATH GPU_IDS=$GPU_IDS SD35_NUM_SAMPLES=$SD35_NUM_SAMPLES"
wait_for_download
selected_gpu="$(wait_for_gpu)"
run_sd35 "$selected_gpu"
