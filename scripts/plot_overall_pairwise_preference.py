import argparse
import csv
import textwrap
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


def font(size: int, bold: bool = False):
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except Exception:
        return ImageFont.load_default()


def load_study(spec: str) -> list[dict]:
    if "=" not in spec:
        raise ValueError(f"Expected LABEL=STUDY_DIR, got {spec}")
    label, study_dir_text = spec.split("=", 1)
    study_dir = Path(study_dir_text)
    rows = read_csv(study_dir / "analysis" / "pairwise_summary.csv")
    by_method = {row["method"]: row for row in rows}
    output = []
    for method in ["sld", "geoclean"]:
        row = by_method[method]
        output.append(
            {
                "setting": label,
                "method": method,
                "wins": row["wins"],
                "trials": row["trials"],
                "win_rate_percent": row["win_rate_percent"],
            }
        )
    return output


def draw_chart(rows: list[dict], output_path: Path) -> None:
    settings = []
    for row in rows:
        if row["setting"] not in settings:
            settings.append(row["setting"])
    by_key = {(row["setting"], row["method"]): float(row["win_rate_percent"]) for row in rows}
    width, height = 820, 460
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((36, 22), "Blind Pairwise Preference: SLD vs GeoClean", fill=(0, 0, 0), font=font(24, True))
    left, top, chart_w, chart_h = 86, 82, 680, 260
    draw.text((34, 58), "Win rate (%)", fill=(0, 0, 0), font=font(13))
    for tick in range(0, 101, 20):
        y = top + chart_h - chart_h * tick / 100
        draw.line([left, y, left + chart_w, y], fill=(225, 225, 225))
        draw.text((34, y - 8), f"{tick}", fill=(70, 70, 70), font=font(13))
    colors = {"sld": (230, 150, 70), "geoclean": (85, 175, 120)}
    names = {"sld": "SLD", "geoclean": "GeoClean"}
    group_w = chart_w / max(1, len(settings))
    bar_w = min(76, group_w * 0.28)
    for group_idx, setting in enumerate(settings):
        center = left + group_w * group_idx + group_w / 2
        for method_idx, method in enumerate(["sld", "geoclean"]):
            pct = by_key.get((setting, method), 0.0)
            x0 = center + (method_idx - 0.5) * (bar_w + 8)
            y0 = top + chart_h - chart_h * pct / 100.0
            draw.rectangle([x0, y0, x0 + bar_w, top + chart_h], fill=colors[method])
            draw.text((x0 + 4, y0 - 22), f"{pct:.0f}%", fill=(0, 0, 0), font=font(15, True))
        wrapped = textwrap.wrap(setting, width=18)
        for line_idx, line in enumerate(wrapped[:2]):
            draw.text((center - 82, top + chart_h + 16 + 17 * line_idx), line, fill=(0, 0, 0), font=font(14))
    legend_x = left + 16
    legend_y = height - 54
    for idx, method in enumerate(["sld", "geoclean"]):
        x = legend_x + idx * 132
        draw.rectangle([x, legend_y, x + 18, legend_y + 18], fill=colors[method])
        draw.text((x + 26, legend_y - 1), names[method], fill=(0, 0, 0), font=font(15))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot overall SLD-vs-GeoClean pairwise preference across settings.")
    parser.add_argument("--study", action="append", required=True, help="LABEL=STUDY_DIR")
    parser.add_argument("--output_dir", default="results/pairwise_preference_overall")
    args = parser.parse_args()

    rows = []
    for spec in args.study:
        rows.extend(load_study(spec))
    output_dir = Path(args.output_dir)
    write_csv(output_dir / "overall_pairwise_summary.csv", rows, ["setting", "method", "wins", "trials", "win_rate_percent"])
    draw_chart(rows, output_dir / "overall_pairwise_bar.png")
    print(f"Wrote {output_dir / 'overall_pairwise_summary.csv'}")
    print(f"Wrote {output_dir / 'overall_pairwise_bar.png'}")


if __name__ == "__main__":
    main()
