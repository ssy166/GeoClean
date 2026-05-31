import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path


RATING_COLUMNS = [
    "unsafe_nudity",
    "visual_quality_1_5",
    "prompt_alignment_1_5",
    "over_suppression_1_5",
]


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_float(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def bootstrap_ci(values: list[float], seed: int = 1023, rounds: int = 2000) -> tuple[float | None, float | None]:
    if not values:
        return None, None
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    boot = []
    for _ in range(rounds):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    boot.sort()
    return boot[int(0.025 * rounds)], boot[int(0.975 * rounds)]


def fmt(value: float | None, scale: float = 1.0) -> str:
    if value is None:
        return ""
    return f"{value * scale:.3f}"


def aggregate(rows: list[dict], group_fields: list[str]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        groups[key].append(row)

    output = []
    for key, items in sorted(groups.items()):
        out = {field: value for field, value in zip(group_fields, key)}
        out["items"] = len(items)
        out["ratings"] = sum(int(item["rating_count"]) for item in items)
        for column in RATING_COLUMNS:
            values = [float(item[column]) for item in items if item[column] != ""]
            value_mean = mean(values)
            lo, hi = bootstrap_ci(values)
            scale = 100.0 if column == "unsafe_nudity" else 1.0
            name = "unsafe_percent" if column == "unsafe_nudity" else column
            out[name] = fmt(value_mean, scale)
            out[f"{name}_ci95"] = (
                f"[{fmt(lo, scale)}, {fmt(hi, scale)}]" if lo is not None and hi is not None else ""
            )
        output.append(out)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize completed GeoClean human-study ratings.")
    parser.add_argument("--study_dir", default="results/human_study_500")
    parser.add_argument("--ratings", nargs="+", required=True, help="Completed rater CSV files.")
    parser.add_argument("--output_dir", default=None)
    args = parser.parse_args()

    study_dir = Path(args.study_dir)
    output_dir = Path(args.output_dir) if args.output_dir else study_dir / "analysis"
    key = {row["item_id"]: row for row in read_csv(study_dir / "study_key.csv")}

    by_item = defaultdict(list)
    long_rows = []
    for rating_path in args.ratings:
        path = Path(rating_path)
        rater = path.stem.replace("_completed", "")
        for row in read_csv(path):
            item_id = row["item_id"]
            if item_id not in key:
                raise KeyError(f"{item_id} from {path} not found in study_key.csv")
            parsed = {column: parse_float(row.get(column, "")) for column in RATING_COLUMNS}
            if all(value is None for value in parsed.values()):
                continue
            item_meta = key[item_id]
            long_row = {
                "rater": rater,
                "item_id": item_id,
                "task": item_meta["task"],
                "model": item_meta["model"],
                "dataset": item_meta["dataset"],
                "method": item_meta["method"],
            }
            for column, value in parsed.items():
                long_row[column] = "" if value is None else value
            long_rows.append(long_row)
            by_item[item_id].append(parsed)

    item_rows = []
    for item_id, ratings in sorted(by_item.items()):
        item_meta = key[item_id]
        item_row = {
            "item_id": item_id,
            "task": item_meta["task"],
            "model": item_meta["model"],
            "dataset": item_meta["dataset"],
            "method": item_meta["method"],
            "rating_count": len(ratings),
        }
        for column in RATING_COLUMNS:
            values = [rating[column] for rating in ratings if rating[column] is not None]
            item_row[column] = "" if not values else f"{mean(values):.6f}"
        item_rows.append(item_row)

    write_csv(
        output_dir / "ratings_long.csv",
        long_rows,
        ["rater", "item_id", "task", "model", "dataset", "method", *RATING_COLUMNS],
    )
    write_csv(
        output_dir / "item_mean_ratings.csv",
        item_rows,
        ["item_id", "task", "model", "dataset", "method", "rating_count", *RATING_COLUMNS],
    )

    summary_fields = [
        "items",
        "ratings",
        "unsafe_percent",
        "unsafe_percent_ci95",
        "visual_quality_1_5",
        "visual_quality_1_5_ci95",
        "prompt_alignment_1_5",
        "prompt_alignment_1_5_ci95",
        "over_suppression_1_5",
        "over_suppression_1_5_ci95",
    ]
    for name, fields in [
        ("summary_by_method.csv", ["method"]),
        ("summary_by_task_method.csv", ["task", "method"]),
        ("summary_by_model_dataset_method.csv", ["model", "dataset", "method"]),
    ]:
        rows = aggregate(item_rows, fields)
        write_csv(output_dir / name, rows, [*fields, *summary_fields])

    print(f"Loaded {len(long_rows)} ratings for {len(item_rows)} items.")
    print(f"Wrote summaries to {output_dir}")


if __name__ == "__main__":
    main()
