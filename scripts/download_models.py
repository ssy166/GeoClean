import argparse
import os

from huggingface_hub import snapshot_download


MODEL_SPECS = {
    "flux": {
        "repo_id": "black-forest-labs/FLUX.1-dev",
        "local_dir": "models/FLUX.1-dev",
        "ignore_patterns": ["flux1-dev.safetensors"],
    },
    "sd35": {
        "repo_id": "stabilityai/stable-diffusion-3.5-medium",
        "local_dir": "models/stable-diffusion-3.5-medium",
        "ignore_patterns": ["sd3.5_medium*.safetensors"],
    },
}


def build_parser():
    parser = argparse.ArgumentParser(description="Download model snapshots needed for GeoClean smoke tests.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=sorted(MODEL_SPECS),
        default=sorted(MODEL_SPECS),
        help="Model snapshots to download.",
    )
    parser.add_argument("--cache_dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--resume", action="store_true", help="Resume partially downloaded snapshots when supported.")
    return parser


def main():
    args = build_parser().parse_args()
    os.makedirs("models", exist_ok=True)

    for name in args.models:
        spec = MODEL_SPECS[name]
        repo_id = spec["repo_id"]
        local_dir = spec["local_dir"]
        print(f"Downloading {repo_id} -> {local_dir}", flush=True)
        snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            cache_dir=args.cache_dir,
            ignore_patterns=spec["ignore_patterns"],
            resume_download=args.resume,
        )
        print(f"Finished {name}: {local_dir}", flush=True)


if __name__ == "__main__":
    main()
