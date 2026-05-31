#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

WORKER="${1:?Usage: run_flux_rebuttal_worker.sh <worker-name>}"
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

OUT_DIR="$ROOT/results/rebuttal_flux"
mkdir -p "$OUT_DIR" "$ROOT/data/rebuttal"

"$PYTHON_BIN" scripts/make_rebuttal_subsets.py \
  --input data/i2p_benchmark.csv \
  --output_dir data/rebuttal \
  --limit "$BROAD_SAMPLES"

run_flux() {
  local name="$1"
  shift
  echo "[$(date -Is)] START $name on CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
  echo "[$(date -Is)] PYTHON=$PYTHON_BIN MODEL=$MODEL_PATH OUT=$OUT_DIR/$name"
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
  echo "[$(date -Is)] DONE $name"
}

case "$WORKER" in
  gpu0_controls_and_preservation)
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
    ;;

  gpu1_pareto_and_early0)
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
    ;;

  gpu2_pareto_and_early2)
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
    ;;

  gpu3_pareto_and_early5)
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
    ;;

  gpu4_pareto_and_early8)
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
    ;;

  gpu5_pareto_and_broad)
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
      --csv_path "data/rebuttal/i2p_violence_${BROAD_SAMPLES}.csv" \
      --prompt_col prompt \
      --num_samples "$BROAD_SAMPLES" \
      --methods base sld geoclean \
      --concept "violence, blood, weapons" \
      --safety_scale 1.5 \
      --sld_threshold 0.0 \
      --acs_delta 0.2

    run_flux broad_i2p_self_harm_s15_full \
      --csv_path "data/rebuttal/i2p_self_harm_${BROAD_SAMPLES}.csv" \
      --prompt_col prompt \
      --num_samples "$BROAD_SAMPLES" \
      --methods base sld geoclean \
      --concept "self-harm, suicide, suffering" \
      --safety_scale 1.5 \
      --sld_threshold 0.0 \
      --acs_delta 0.2

    run_flux broad_i2p_illegal_s15_full \
      --csv_path "data/rebuttal/i2p_illegal_activity_${BROAD_SAMPLES}.csv" \
      --prompt_col prompt \
      --num_samples "$BROAD_SAMPLES" \
      --methods base sld geoclean \
      --concept "illegal activity, drug use, theft, vandalism" \
      --safety_scale 1.5 \
      --sld_threshold 0.0 \
      --acs_delta 0.2
    ;;

  *)
    echo "Unknown worker '$WORKER'" >&2
    exit 2
    ;;
esac
