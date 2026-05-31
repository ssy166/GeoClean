#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-$ROOT/.venv/bin/python}"
MODEL_PATH="${MODEL_PATH:-models/FLUX.1-dev}"
NUM_SAMPLES="${NUM_SAMPLES:-100}"
BROAD_SAMPLES="${BROAD_SAMPLES:-50}"
STEPS="${STEPS:-28}"
HEIGHT="${HEIGHT:-512}"
WIDTH="${WIDTH:-512}"
GUIDANCE_SCALE="${GUIDANCE_SCALE:-3.5}"
SEED="${SEED:-42}"
DTYPE="${DTYPE:-bfloat16}"

LOG_DIR="$ROOT/logs/rebuttal_flux"
OUT_DIR="$ROOT/results/rebuttal_flux"
mkdir -p "$LOG_DIR" "$OUT_DIR" "$ROOT/data/rebuttal"

"$PYTHON_BIN" scripts/make_rebuttal_subsets.py \
  --input data/i2p_benchmark.csv \
  --output_dir data/rebuttal \
  --limit "$BROAD_SAMPLES"

run_flux() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[$(date -Is)] PYTHON=$PYTHON_BIN MODEL=$MODEL_PATH OUT=$OUT_DIR/$name"
  set +e
  "$PYTHON_BIN" sample/baseline_flux.py \
    --model_path "$MODEL_PATH" \
    --height "$HEIGHT" \
    --width "$WIDTH" \
    --num_inference_steps "$STEPS" \
    --guidance_scale "$GUIDANCE_SCALE" \
    --seed "$SEED" \
    --dtype "$DTYPE" \
    --device cuda \
    --output_dir "$OUT_DIR/$name" \
    "$@"
  local status="$?"
  set -e
  echo "[$(date -Is)] EXIT $name status=$status"
  if [[ "$status" -ne 0 ]]; then
    return "$status"
  fi
  echo "[$(date -Is)] DONE $name"
}

export PYTHON_BIN MODEL_PATH NUM_SAMPLES BROAD_SAMPLES STEPS HEIGHT WIDTH GUIDANCE_SCALE SEED DTYPE
export OUT_DIR
export -f run_flux

launch_worker() {
  local gpu="$1"
  local name="$2"
  shift 2
  CUDA_VISIBLE_DEVICES="$gpu" nohup "$@" > "$LOG_DIR/${name}.log" 2>&1 &
  local pid="$!"
  echo "$pid $gpu $name" | tee -a "$LOG_DIR/pids.txt"
}

: > "$LOG_DIR/pids.txt"

launch_worker 0 gpu0_controls_and_preservation bash -c '
  run_flux controlled_i2p_s15_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods base np ca dve sld geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux preservation_mma_clean_s15_full \
    --csv_path data/MMA.csv \
    --prompt_col clean_prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods base sld geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2
'

launch_worker 1 gpu1_pareto_and_early0 bash -c '
  run_flux pareto_i2p_s10_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods sld geoclean \
    --concept nudity \
    --safety_scale 1.0 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux earlystop_i2p_s15_threshold00 \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2
'

launch_worker 2 gpu2_pareto_and_early2 bash -c '
  run_flux pareto_i2p_s15_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods sld geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux earlystop_i2p_s15_threshold02 \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.2 \
    --acs_delta 0.2
'

launch_worker 3 gpu3_pareto_and_early5 bash -c '
  run_flux pareto_i2p_s20_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods sld geoclean \
    --concept nudity \
    --safety_scale 2.0 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux earlystop_i2p_s15_threshold05 \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.5 \
    --acs_delta 0.2
'

launch_worker 4 gpu4_pareto_and_early8 bash -c '
  run_flux pareto_i2p_s25_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods sld geoclean \
    --concept nudity \
    --safety_scale 2.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux earlystop_i2p_s15_threshold08 \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods geoclean \
    --concept nudity \
    --safety_scale 1.5 \
    --sld_threshold 0.8 \
    --acs_delta 0.2
'

launch_worker 5 gpu5_pareto_and_broad bash -c '
  run_flux pareto_i2p_s30_full \
    --csv_path data/i2p_benchmark.csv \
    --prompt_col prompt \
    --num_samples "$NUM_SAMPLES" \
    --methods sld geoclean \
    --concept nudity \
    --safety_scale 3.0 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux broad_i2p_violence_s15_full \
    --csv_path data/rebuttal/i2p_violence_'"$BROAD_SAMPLES"'.csv \
    --prompt_col prompt \
    --num_samples "$BROAD_SAMPLES" \
    --methods base sld geoclean \
    --concept "violence, blood, weapons" \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux broad_i2p_self_harm_s15_full \
    --csv_path data/rebuttal/i2p_self_harm_'"$BROAD_SAMPLES"'.csv \
    --prompt_col prompt \
    --num_samples "$BROAD_SAMPLES" \
    --methods base sld geoclean \
    --concept "self-harm, suicide, suffering" \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2

  run_flux broad_i2p_illegal_s15_full \
    --csv_path data/rebuttal/i2p_illegal_activity_'"$BROAD_SAMPLES"'.csv \
    --prompt_col prompt \
    --num_samples "$BROAD_SAMPLES" \
    --methods base sld geoclean \
    --concept "illegal activity, drug use, theft, vandalism" \
    --safety_scale 1.5 \
    --sld_threshold 0.0 \
    --acs_delta 0.2
'

echo "Launched FLUX rebuttal workers. Logs: $LOG_DIR"
cat "$LOG_DIR/pids.txt"
