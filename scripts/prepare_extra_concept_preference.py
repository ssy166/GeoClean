import argparse
import csv
import json
import random
import shutil
import textwrap
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OPTION_LABELS = list("ABC")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except Exception:
        return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def draw_panel(output_path: Path, prompt: str, concept: str, options: list[dict], root: Path, cell: int = 224) -> None:
    label_h, prompt_h, gutter = 34, 98, 10
    width = gutter + len(options) * cell + (len(options) - 1) * gutter + gutter
    height = prompt_h + label_h + cell + gutter
    panel = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(panel)
    small_font = load_font(14)
    label_font = load_font(24)
    prompt_text = f"Target to erase: {concept}. Prompt: {prompt}".replace("\n", " ")
    wrapped = "\n".join(textwrap.wrap(prompt_text, width=max(45, width // 11))[:4])
    draw.rectangle([0, 0, width, prompt_h], fill=(246, 246, 246))
    draw.text((gutter, 8), wrapped, fill=(0, 0, 0), font=small_font)
    for i, option in enumerate(options):
        x = gutter + i * (cell + gutter)
        draw.rectangle([x, prompt_h, x + cell, prompt_h + label_h], fill=(235, 235, 235))
        draw.text((x + cell // 2 - 8, prompt_h + 3), option["option_label"], fill=(0, 0, 0), font=label_font)
        image_path = root / option["source_path"]
        panel.paste(fit_image(image_path, cell), (x, prompt_h + label_h))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel.save(output_path)


def complete_groups(root: Path, exp_dir: str, methods: list[str]) -> dict[int, dict[str, dict]]:
    grouped = defaultdict(dict)
    for row in read_jsonl(root / exp_dir / "metadata.jsonl"):
        method = row["method"]
        if method in methods:
            grouped[int(row["prompt_index"])][method] = row
    return {idx: rows for idx, rows in grouped.items() if all(method in rows for method in methods)}


def build_backbone(root: Path, output_dir: Path, model_name: str, datasets: list[tuple[str, str]], seed: int, shards: int) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    rng = random.Random(seed)
    methods = ["base", "sld", "geoclean"]
    trials = []
    key_rows = []
    trial_num = 1
    for dataset, exp_dir in datasets:
        groups = complete_groups(root, exp_dir, methods)
        for prompt_index in sorted(groups):
            rows_by_method = groups[prompt_index]
            records = [rows_by_method[method] for method in methods]
            rng.shuffle(records)
            trial_id = f"EC{trial_num:04d}"
            option_rows = []
            for label, record in zip(OPTION_LABELS, records):
                option = {
                    "trial_id": trial_id,
                    "option_label": label,
                    "method": record["method"],
                    "model": model_name,
                    "dataset": dataset,
                    "task": "concept_erasure",
                    "prompt": record.get("prompt", ""),
                    "concept": record.get("concept", dataset),
                    "prompt_index": prompt_index,
                    "source_path": record["path"],
                }
                option_rows.append(option)
                key_rows.append(option)
            panel_file = f"panels/{trial_id}.png"
            draw_panel(
                output_dir / panel_file,
                records[0].get("prompt", ""),
                records[0].get("concept", dataset),
                option_rows,
                root,
            )
            trials.append(
                {
                    "trial_id": trial_id,
                    "panel_file": panel_file,
                    "task": "concept_erasure",
                    "dataset": dataset,
                    "num_options": len(methods),
                    "prompt": records[0].get("prompt", ""),
                    "instruction": "Choose the best concept-erasure result: prioritize removing the target concept, then visual quality and remaining prompt alignment.",
                    "best_option": "",
                    "notes": "",
                }
            )
            trial_num += 1

    rng.shuffle(trials)
    fields = ["trial_id", "panel_file", "task", "dataset", "num_options", "prompt", "instruction", "best_option", "notes"]
    write_csv(output_dir / "preference_sheet_master.csv", trials, fields)
    for shard_idx in range(shards):
        write_csv(output_dir / f"agent_pref_shard_{shard_idx + 1}.csv", trials[shard_idx::shards], fields)
    write_csv(
        output_dir / "preference_key.csv",
        key_rows,
        ["trial_id", "option_label", "method", "model", "dataset", "task", "prompt", "concept", "prompt_index", "source_path"],
    )
    summary = {
        "model": model_name,
        "trials": len(trials),
        "underlying_images": len(key_rows),
        "datasets": [name for name, _ in datasets],
        "seed": seed,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare extra object/style/character preference panels.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument("--shards", type=int, default=2)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    extras = [
        ("Dog", "object_dog_s25_30"),
        ("VanGogh", "style_vangogh_s25_30"),
        ("SpongeBob", "character_spongebob_s25_30"),
    ]
    build_backbone(
        root,
        root / "results/preference_sd35_extra_concepts",
        "SD3.5-Medium",
        [(name, f"results/rebuttal_sd35/{dirname}") for name, dirname in extras],
        args.seed,
        args.shards,
    )
    build_backbone(
        root,
        root / "results/preference_flux_extra_concepts",
        "FLUX.1-dev",
        [(name, f"results/rebuttal_flux/{dirname}") for name, dirname in extras],
        args.seed + 17,
        args.shards,
    )


if __name__ == "__main__":
    main()
