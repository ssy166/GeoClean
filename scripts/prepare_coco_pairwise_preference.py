import argparse
import csv
import json
import random
import shutil
import textwrap
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


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


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def fit_image(path: Path, size: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2))
    return canvas


def draw_panel(path: Path, prompt: str, options: list[dict], root: Path) -> None:
    cell = 300
    gutter = 12
    prompt_h = 82
    label_h = 38
    width = gutter + 2 * cell + gutter + gutter
    height = prompt_h + label_h + cell + gutter
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle([0, 0, width, prompt_h], fill=(246, 246, 246))
    wrapped = "\n".join(textwrap.wrap("Prompt: " + prompt.replace("\n", " "), width=82)[:3])
    draw.text((gutter, 8), wrapped, fill=(0, 0, 0), font=font(16))
    for idx, option in enumerate(options):
        x = gutter + idx * (cell + gutter)
        draw.rectangle([x, prompt_h, x + cell, prompt_h + label_h], fill=(235, 235, 235))
        draw.text((x + cell // 2 - 10, prompt_h + 4), option["option_label"], fill=(0, 0, 0), font=font(26, True))
        image.paste(fit_image(root / option["path"], cell), (x, prompt_h + label_h))
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare 100 MSCOCO SLD-vs-GeoClean pairwise preference panels.")
    parser.add_argument("--root", default=".")
    parser.add_argument("--sld_dir", default="results/rebuttal_sd35/coco_quality500_sld")
    parser.add_argument("--geoclean_dir", default="results/rebuttal_sd35/coco_quality500_geoclean")
    parser.add_argument("--output_dir", default="results/coco_pairwise_preference_100")
    parser.add_argument("--num_prompts", type=int, default=100)
    parser.add_argument("--num_shards", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1023)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    out = root / args.output_dir
    if out.exists():
        shutil.rmtree(out)
    panels = out / "panels"
    rng = random.Random(args.seed)

    sld = {int(r["prompt_index"]): r for r in read_jsonl(root / args.sld_dir / "metadata.jsonl")}
    geo = {int(r["prompt_index"]): r for r in read_jsonl(root / args.geoclean_dir / "metadata.jsonl")}
    common = sorted(set(sld) & set(geo))
    selected = sorted(rng.sample(common, args.num_prompts))

    sheet_rows = []
    key_rows = []
    for trial_idx, prompt_index in enumerate(selected, start=1):
        trial_id = f"CP{trial_idx:04d}"
        pair = [
            {"method": "sld", **sld[prompt_index]},
            {"method": "geoclean", **geo[prompt_index]},
        ]
        rng.shuffle(pair)
        options = []
        for label, record in zip(["A", "B"], pair):
            option = {
                "trial_id": trial_id,
                "option_label": label,
                "method": record["method"],
                "prompt_index": prompt_index,
                "prompt": record["prompt"],
                "path": record["path"],
            }
            options.append(option)
            key_rows.append(option)
        panel_file = f"panels/{trial_id}.png"
        draw_panel(root / args.output_dir / panel_file, sld[prompt_index]["prompt"], options, root)
        sheet_rows.append(
            {
                "trial_id": trial_id,
                "panel_file": panel_file,
                "prompt_index": prompt_index,
                "prompt": sld[prompt_index]["prompt"],
                "instruction": "Choose the better preserved image: prioritize prompt alignment and visual quality; penalize artifacts or unnecessary content loss.",
                "best_option": "",
                "notes": "",
            }
        )

    rng.shuffle(sheet_rows)
    fields = ["trial_id", "panel_file", "prompt_index", "prompt", "instruction", "best_option", "notes"]
    write_csv(out / "pairwise_sheet_master.csv", sheet_rows, fields)
    for shard_idx in range(args.num_shards):
        write_csv(out / f"agent_pairwise_shard_{shard_idx + 1}.csv", sheet_rows[shard_idx:: args.num_shards], fields)
    write_csv(out / "pairwise_key.csv", key_rows, ["trial_id", "option_label", "method", "prompt_index", "prompt", "path"])
    (out / "README.md").write_text(
        "# MSCOCO Pairwise Preference 100\n\n"
        "100 matched MSCOCO prompts comparing SLD vs GeoClean under default outputs. "
        "Panels are anonymized as A/B; do not show pairwise_key.csv to raters.\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(sheet_rows)} pairwise trials to {out}")


if __name__ == "__main__":
    main()
