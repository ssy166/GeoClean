import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


METHODS = ["base", "dve", "sld", "geoclean"]
METHOD_LABELS = {
    "base": "Base",
    "dve": "DVE",
    "sld": "SLD",
    "geoclean": "GeoClean",
}
DISPLAY_LABELS = {
    "adversarial_safety": "Adversarial",
    "benign_preservation": "Benign",
    "COCO-human500": "COCO",
    "MMA-Adv": "MMA-Adv",
    "P4D": "P4D",
    "Ring16": "Ring16",
    "Ring38": "Ring38",
    "Ring77": "Ring77",
    "UnDiff": "UnDiff",
    "I2P": "I2P",
    "MMA-clean": "MMA-clean",
}
COLORS = {
    "base": (145, 145, 145),
    "dve": (110, 150, 210),
    "sld": (230, 150, 70),
    "geoclean": (85, 175, 120),
}


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            pass
    return ImageFont.load_default()


def normalize_choice(value: str) -> str:
    value = (value or "").strip().upper()
    return value[:1] if value[:1] in {"A", "B", "C", "D"} else ""


def summarize(completed_paths: list[Path], key_path: Path, output_dir: Path) -> tuple[list[dict], list[dict]]:
    key = {(row["trial_id"], row["option_label"]): row for row in read_csv(key_path)}
    long_rows = []
    errors = []
    for path in completed_paths:
        rater = path.stem.replace("_completed", "")
        for row in read_csv(path):
            trial_id = row["trial_id"]
            choice = normalize_choice(row.get("best_option", ""))
            if not choice:
                errors.append({"rater": rater, "trial_id": trial_id, "error": "missing_or_invalid_choice"})
                continue
            chosen = key.get((trial_id, choice))
            if chosen is None:
                errors.append({"rater": rater, "trial_id": trial_id, "error": f"choice_not_in_key:{choice}"})
                continue
            long_rows.append(
                {
                    "rater": rater,
                    "trial_id": trial_id,
                    "task": chosen["task"],
                    "dataset": chosen["dataset"],
                    "method": chosen["method"],
                    "choice": choice,
                }
            )

    write_csv(output_dir / "preference_long.csv", long_rows, ["rater", "trial_id", "task", "dataset", "method", "choice"])
    if errors:
        write_csv(output_dir / "preference_errors.csv", errors, ["rater", "trial_id", "error"])

    summaries = []
    for group_fields, filename in [
        (["task"], "preference_by_task.csv"),
        (["dataset"], "preference_by_dataset.csv"),
        (["task", "dataset"], "preference_by_task_dataset.csv"),
    ]:
        groups = defaultdict(list)
        for row in long_rows:
            groups[tuple(row[field] for field in group_fields)].append(row)
        rows = []
        for group_key, items in sorted(groups.items()):
            counts = Counter(row["method"] for row in items)
            total = len(items)
            for method in METHODS:
                if counts.get(method, 0) == 0 and method == "dve":
                    continue
                out = {field: value for field, value in zip(group_fields, group_key)}
                out.update(
                    {
                        "method": method,
                        "wins": counts.get(method, 0),
                        "trials": total,
                        "win_rate_percent": f"{100.0 * counts.get(method, 0) / total:.2f}" if total else "0.00",
                    }
                )
                rows.append(out)
                summaries.append({**out, "group": filename})
        write_csv(output_dir / filename, rows, [*group_fields, "method", "wins", "trials", "win_rate_percent"])
    return long_rows, summaries


def draw_chart(csv_path: Path, output_path: Path, title: str, group_field: str) -> None:
    rows = read_csv(csv_path)
    groups = []
    for row in rows:
        group = row[group_field]
        if group not in groups:
            groups.append(group)
    methods = [method for method in METHODS if any(row["method"] == method for row in rows)]

    width = max(900, 130 * len(groups) + 180)
    height = 520
    left, right, top, bottom = 80, 30, 58, 115
    chart_w = width - left - right
    chart_h = height - top - bottom
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    title_font = load_font(24, True)
    font = load_font(15)
    small = load_font(12)
    draw.text((left, 16), title, fill=(0, 0, 0), font=title_font)
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        draw.line([left, y, width - right, y], fill=(225, 225, 225))
        draw.text((25, y - 8), str(tick), fill=(80, 80, 80), font=small)
    draw.line([left, top, left, top + chart_h], fill=(0, 0, 0))
    draw.line([left, top + chart_h, width - right, top + chart_h], fill=(0, 0, 0))

    data = {(row[group_field], row["method"]): float(row["win_rate_percent"]) for row in rows}
    group_w = chart_w / max(1, len(groups))
    bar_w = min(22, group_w / (len(methods) + 1.5))
    for gi, group in enumerate(groups):
        center = left + gi * group_w + group_w / 2
        start = center - (len(methods) * bar_w + (len(methods) - 1) * 4) / 2
        for mi, method in enumerate(methods):
            value = data.get((group, method), 0.0)
            x0 = start + mi * (bar_w + 4)
            y0 = top + chart_h - chart_h * value / 100.0
            draw.rectangle([x0, y0, x0 + bar_w, top + chart_h], fill=COLORS[method])
            if value >= 5:
                draw.text((x0 - 2, y0 - 16), f"{value:.0f}", fill=(0, 0, 0), font=small)
        label = DISPLAY_LABELS.get(group, group)
        bbox = draw.textbbox((0, 0), label, font=small)
        draw.text((center - (bbox[2] - bbox[0]) / 2, top + chart_h + 10), label, fill=(0, 0, 0), font=small)

    legend_x = left
    legend_y = height - 38
    for method in methods:
        draw.rectangle([legend_x, legend_y, legend_x + 15, legend_y + 15], fill=COLORS[method])
        draw.text((legend_x + 20, legend_y - 1), METHOD_LABELS[method], fill=(0, 0, 0), font=font)
        legend_x += 110
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize matched preference study choices.")
    parser.add_argument("--study_dir", default="results/preference_study_500")
    parser.add_argument("--completed", nargs="+", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--title_prefix", default="")
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    output_dir = Path(args.output_dir) if args.output_dir else study_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    completed_paths = [Path(path) for path in args.completed]
    long_rows, _ = summarize(completed_paths, study_dir / "preference_key.csv", output_dir)
    prefix = f"{args.title_prefix} " if args.title_prefix else ""
    draw_chart(output_dir / "preference_by_dataset.csv", output_dir / "preference_by_dataset.png", f"{prefix}Preference Win Rate by Dataset", "dataset")
    draw_chart(output_dir / "preference_by_task.csv", output_dir / "preference_by_task.png", f"{prefix}Preference Win Rate by Task", "task")
    print(f"Loaded {len(long_rows)} choices.")
    print(f"Wrote summaries and charts to {output_dir}")


if __name__ == "__main__":
    main()
