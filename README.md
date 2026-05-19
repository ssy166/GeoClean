# GeoClean

**Training-free concept erasure for rectified-flow text-to-image models.**

GeoClean is the cleaned public release of the verified FLUX main workflow. The
repository intentionally keeps only the code path used by the paper figures and
tables: the sampler, the FLUX loader, prompt datasets, the runtime requirement
file, and the project page.

Project page: <https://ssy166.github.io/GeoClean/>

## Repository Layout

```text
sample/SLD_FLUX.py                         # Main SLD / CLE / ACS / GeoClean sampler
utils/model_loader.py                      # Minimal local FLUX pipeline loader
data/i2p_benchmark.csv                     # I2P benchmark prompts
data/MMA.csv                               # MMA prompt set
data/ring16.csv                            # Ring-16 prompt set
data/ring38.csv                            # Ring-38 prompt set
data/ring77.csv                            # Ring-77 prompt set
data/P4D.csv                               # P4D prompt set
data/UnDiff.csv                            # UnDiff prompt set
environments/cfg/requirements_flux_runtime.txt
docs/                                      # GitHub Pages project website
```

Not included: model weights, generated images, W&B logging helpers, historical
batch runners, notebooks, local environments, and evaluation caches.

## Setup

Use Python 3.10 and install the FLUX runtime stack:

```bash
pip install -r environments/cfg/requirements_flux_runtime.txt
```

Place FLUX weights outside Git-tracked files:

```text
models/FLUX.1-dev
```

## Run GeoClean

```bash
python sample/SLD_FLUX.py \
  --csv_path data/i2p_benchmark.csv \
  --use_sld \
  --use_cle \
  --use_acs \
  --safety_scale 1.5 \
  --acs_delta 0.2 \
  --guidance_scale 3.5 \
  --safety_concept nudity
```

Outputs are saved locally under `results/`. No cloud logging or hidden upload
path is used in this public main workflow.

## Run Included Baselines

The baseline runner exposes one CLI entry for each baseline implemented in this
repository:

```bash
python sample/baseline_flux.py \
  --model_path models/FLUX.1-dev \
  --csv_path data/i2p_benchmark.csv \
  --methods base np ca dev sld geoclean \
  --num_samples 1 \
  --num_inference_steps 28 \
  --concept nudity
```

The runner only exposes baselines with actual code in this repository:
`base`, `np`, `ca`, `dev`, `sld`, and `geoclean`.

## Verified Smoke Results

The public baseline runner was smoke-tested on the SenseTime server with the
same temporary FLUX runtime environment used during repository cleanup:

```text
workdir: /mnt/afs/intern/manlichen/ssy/GeoClean_baseline_smoke
env: /mnt/afs/intern/manlichen/ssy/conda_envs/geoclean_flux_tmp
model: /mnt/afs/intern/manlichen/ssy/downloads/models/Niansuh-FLUX.1-schnell
output: results/baseline_smoke_4step_real
```

Smoke command:

```bash
python sample/baseline_flux.py \
  --model_path /mnt/afs/intern/manlichen/ssy/downloads/models/Niansuh-FLUX.1-schnell \
  --csv_path data/i2p_benchmark.csv \
  --output_dir results/baseline_smoke_4step_real \
  --methods base np ca dev sld geoclean \
  --num_samples 1 \
  --num_inference_steps 4 \
  --height 512 \
  --width 512 \
  --concept nudity \
  --guidance_scale 0.0 \
  --safety_scale 1.5 \
  --acs_delta 0.2 \
  --seed 123
```

Result: the 6 executable inference-time baselines completed and produced one
`512x512` PNG each: `base`, `np`, `ca`, `dev`, `sld`, and `geoclean`.
`metadata.jsonl` contains one record per method with the prompt, seed, saved
path, and adapter note. This is a runtime smoke test, not the full paper-table
metric reproduction. Full metric reproduction still requires FLUX.1-dev,
paper-scale prompt counts, and NudeNet/CLIP/FID evaluation.

Dataset prompt columns:

```text
i2p_benchmark.csv, ring16.csv, ring38.csv, ring77.csv, P4D.csv, UnDiff.csv -> prompt
MMA.csv -> adv_prompt or target_prompt
```

## Method Names

- **CLE**: Competition-Aware Lookahead Evaluation, enabled with `--use_cle`.
- **ACS**: Amplification-Controlled Correction Smoothing, enabled with
  `--use_acs --acs_delta <value>`.
- **GeoClean**: full CLE + ACS sampler.

Legacy aliases that still map to the paper modules are kept:
`--use_rectified_sld` for CLE, `--sld_use_gtr3` for ACS, and
`--sld_gtr3_delta` for the ACS threshold.

## Citation

```bibtex
@misc{geoclean2026,
  title  = {GeoClean: Training-Free Concept Erasure in Rectified Flow via Posterior-Competition Stabilization},
  author = {GeoClean Authors},
  year   = {2026}
}
```
