# GeoClean

**Training-free concept erasure for rectified-flow text-to-image models.**

GeoClean is the cleaned public release of the paper-aligned FLUX main workflow. The
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

The baseline runner exposes one method flag for each executable inference-time
baseline included in this repository:

```text
base      -> unmodified FLUX generation
np        -> negative-prompt vector guidance
ca        -> concept-ablation guidance
dev       -> directional erasure vector guidance
sld       -> safe latent diffusion guidance adapted to FLUX
geoclean  -> full CLE + ACS sampler
```

Run all included baselines on the I2P benchmark:

```bash
python sample/baseline_flux.py \
  --model_path models/FLUX.1-dev \
  --csv_path data/i2p_benchmark.csv \
  --methods base np ca dev sld geoclean \
  --num_samples 1 \
  --num_inference_steps 28 \
  --concept nudity
```

Run a single baseline by passing one method name:

```bash
python sample/baseline_flux.py \
  --model_path models/FLUX.1-dev \
  --csv_path data/i2p_benchmark.csv \
  --methods geoclean \
  --num_samples 100 \
  --num_inference_steps 28 \
  --concept nudity \
  --output_dir results/baselines_flux
```

For MMA, use the adversarial prompt column:

```bash
python sample/baseline_flux.py \
  --model_path models/FLUX.1-dev \
  --csv_path data/MMA.csv \
  --prompt_col adv_prompt \
  --methods geoclean \
  --num_samples 100 \
  --num_inference_steps 28 \
  --concept nudity \
  --output_dir results/baselines_mma
```

Outputs are written to the selected `--output_dir`. Use `--methods base`, `np`,
`ca`, `dev`, `sld`, or `geoclean` to reproduce each baseline separately.

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
