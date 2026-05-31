import argparse
import csv
from collections import Counter
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_choice(value: str) -> str:
    value = (value or "").strip().upper()
    return value if value in {"A", "B"} else ""


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def draw_chart(output_path: Path, counts: Counter, total: int, title: str) -> None:
    width, height = 560, 380
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((35, 20), title, fill=(0, 0, 0), font=font(22, True))
    left, top, chart_h = 90, 80, 210
    bar_w = 110
    gap = 70
    colors = {"sld": (230, 150, 70), "geoclean": (85, 175, 120)}
    labels = {"sld": "SLD", "geoclean": "GeoClean"}
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        draw.line([left - 20, y, width - 60, y], fill=(225, 225, 225))
        draw.text((35, y - 8), str(tick), fill=(80, 80, 80), font=font(12))
    for idx, method in enumerate(["sld", "geoclean"]):
        pct = 100.0 * counts[method] / total if total else 0.0
        x0 = left + idx * (bar_w + gap)
        y0 = top + chart_h - chart_h * pct / 100.0
        draw.rectangle([x0, y0, x0 + bar_w, top + chart_h], fill=colors[method])
        draw.text((x0 + 22, y0 - 24), f"{pct:.1f}%", fill=(0, 0, 0), font=font(16, True))
        draw.text((x0 + 24, top + chart_h + 12), labels[method], fill=(0, 0, 0), font=font(16))
    draw.text((left - 50, top + chart_h + 55), "Win rate (%)", fill=(0, 0, 0), font=font(14))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize SLD-vs-GeoClean pairwise choices.")
    parser.add_argument("--study_dir", default="results/coco_pairwise_preference_100")
    parser.add_argument("--completed", nargs="+", required=True)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--chart_title", default="Pairwise Preference")
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    output_dir = Path(args.output_dir) if args.output_dir else study_dir / "analysis"
    key = {(r["trial_id"], r["option_label"]): r for r in read_csv(study_dir / "pairwise_key.csv")}
    long_rows = []
    errors = []
    for path_text in args.completed:
        path = Path(path_text)
        rater = path.stem.replace("_completed", "")
        for row in read_csv(path):
            choice = normalize_choice(row.get("best_option", ""))
            if not choice:
                errors.append({"rater": rater, "trial_id": row["trial_id"], "error": "invalid_choice"})
                continue
            chosen = key.get((row["trial_id"], choice))
            if chosen is None:
                errors.append({"rater": rater, "trial_id": row["trial_id"], "error": f"missing_key:{choice}"})
                continue
            long_rows.append(
                {
                    "rater": rater,
                    "trial_id": row["trial_id"],
                    "prompt_index": chosen["prompt_index"],
                    "choice": choice,
                    "method": chosen["method"],
                }
            )
    counts = Counter(r["method"] for r in long_rows)
    total = len(long_rows)
    summary_rows = [
        {
            "method": method,
            "wins": counts[method],
            "trials": total,
            "win_rate_percent": f"{100.0 * counts[method] / total:.2f}" if total else "0.00",
        }
        for method in ["sld", "geoclean"]
    ]
    write_csv(output_dir / "pairwise_long.csv", long_rows, ["rater", "trial_id", "prompt_index", "choice", "method"])
    write_csv(output_dir / "pairwise_summary.csv", summary_rows, ["method", "wins", "trials", "win_rate_percent"])
    if errors:
        write_csv(output_dir / "pairwise_errors.csv", errors, ["rater", "trial_id", "error"])
    draw_chart(output_dir / "pairwise_preference_bar.png", counts, total, args.chart_title)
    print(f"Loaded {total} choices. GeoClean wins: {counts['geoclean']}; SLD wins: {counts['sld']}")


if __name__ == "__main__":
    main()
