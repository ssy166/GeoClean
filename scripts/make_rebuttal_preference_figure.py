import argparse
import csv
import shutil
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = ["base", "dve", "sld", "geoclean"]
LABELS = {"base": "Base", "dve": "DVE", "sld": "SLD", "geoclean": "GeoClean"}
COLORS = {
    "base": (145, 145, 145),
    "dve": (95, 140, 205),
    "sld": (230, 145, 65),
    "geoclean": (75, 170, 115),
}
DISPLAY = {
    "COCO-human500": "COCO",
    "MMA-Adv": "MMA",
    "MMA-clean": "MMA",
    "VanGogh": "VG",
    "SpongeBob": "SB",
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["dataset", "method", "wins", "trials", "win_rate_percent"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def merge_rows(paths: list[Path], order: list[str]) -> list[dict]:
    rows = []
    for path in paths:
        rows.extend(read_csv(path))
    rank = {name: i for i, name in enumerate(order)}
    rows.sort(key=lambda row: (rank.get(row["dataset"], 999), METHODS.index(row["method"])))
    return rows


def draw_chart(rows: list[dict], output_path: Path, title: str, order: list[str], width: int = 980, height: int = 260) -> None:
    datasets = [name for name in order if any(row["dataset"] == name for row in rows)]
    present_methods = [m for m in METHODS if any(row["method"] == m for row in rows)]
    data = {(row["dataset"], row["method"]): float(row["win_rate_percent"]) for row in rows}

    left, right, top, bottom = 36, 14, 34, 46
    chart_w = width - left - right
    chart_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(18, True)
    font = load_font(12)
    small = load_font(10)
    draw.text((left, 7), title, fill=(0, 0, 0), font=title_font)
    for tick in [0, 50, 100]:
        y = top + chart_h - chart_h * tick / 100
        draw.line([left, y, width - right, y], fill=(226, 226, 226))
        draw.text((9, y - 7), str(tick), fill=(70, 70, 70), font=small)
    draw.line([left, top + chart_h, width - right, top + chart_h], fill=(0, 0, 0))

    group_w = chart_w / max(1, len(datasets))
    bar_w = max(5, min(15, (group_w - 7) / max(1, len(present_methods))))
    gap = 2
    for gi, dataset in enumerate(datasets):
        center = left + gi * group_w + group_w / 2
        start = center - (len(present_methods) * bar_w + (len(present_methods) - 1) * gap) / 2
        for mi, method in enumerate(present_methods):
            if (dataset, method) not in data:
                continue
            value = data[(dataset, method)]
            x0 = start + mi * (bar_w + gap)
            y0 = top + chart_h - chart_h * value / 100.0
            draw.rectangle([x0, y0, x0 + bar_w, top + chart_h], fill=COLORS[method])
            if value >= 5:
                draw.text((x0 - 1, y0 - 12), f"{value:.0f}", fill=(0, 0, 0), font=small)
        label = DISPLAY.get(dataset, dataset)
        bbox = draw.textbbox((0, 0), label, font=small)
        draw.text((center - (bbox[2] - bbox[0]) / 2, top + chart_h + 7), label, fill=(0, 0, 0), font=small)

    x = left
    y = height - 17
    for method in present_methods:
        draw.rectangle([x, y, x + 9, y + 9], fill=COLORS[method])
        draw.text((x + 12, y - 2), LABELS[method], fill=(0, 0, 0), font=small)
        x += 68
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def stack_compact(sd_path: Path, flux_path: Path, output_path: Path) -> None:
    sd = Image.open(sd_path).convert("RGB")
    flux = Image.open(flux_path).convert("RGB")
    target_w = 820
    target_h = 118
    sd = sd.resize((target_w, target_h), Image.Resampling.LANCZOS)
    flux = flux.resize((target_w, target_h), Image.Resampling.LANCZOS)
    out = Image.new("RGB", (target_w, target_h * 2 + 6), "white")
    out.paste(sd, (0, 0))
    out.paste(flux, (0, target_h + 6))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge backbone preference CSVs and render rebuttal charts.")
    parser.add_argument("--root", default=".")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    out_dir = root / "results/preference_figures"
    sd_order = ["COCO-human500", "MMA-Adv", "P4D", "Ring16", "Ring38", "Ring77", "UnDiff", "Dog", "VanGogh", "SpongeBob"]
    flux_order = ["I2P", "MMA-clean", "Ring16", "Ring38", "Ring77", "Dog", "VanGogh", "SpongeBob"]
    sd_rows = merge_rows(
        [
            root / "results/preference_sd35_backbone/analysis/preference_by_dataset.csv",
            root / "results/preference_sd35_extra_concepts/analysis/preference_by_dataset.csv",
        ],
        sd_order,
    )
    flux_rows = merge_rows(
        [
            root / "results/preference_flux_backbone/analysis/preference_by_dataset.csv",
            root / "results/preference_flux_extra_concepts/analysis/preference_by_dataset.csv",
        ],
        flux_order,
    )
    sd_csv = out_dir / "sd35_preference_by_dataset_merged.csv"
    flux_csv = out_dir / "flux_preference_by_dataset_merged.csv"
    sd_png = out_dir / "sd35_preference_by_dataset_merged.png"
    flux_png = out_dir / "flux_preference_by_dataset_merged.png"
    write_csv(sd_csv, sd_rows)
    write_csv(flux_csv, flux_rows)
    draw_chart(sd_rows, sd_png, "SD3.5 Preference (%)", sd_order)
    draw_chart(flux_rows, flux_png, "FLUX Preference (%)", flux_order)
    stack_compact(sd_png, flux_png, root / "author_response_template/fig_pref_both_compact.png")
    shutil.copyfile(sd_png, root / "author_response_template/fig_sd35_pref.png")
    shutil.copyfile(flux_png, root / "author_response_template/fig_flux_pref.png")


if __name__ == "__main__":
    main()
