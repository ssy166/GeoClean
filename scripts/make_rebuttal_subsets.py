import argparse
import csv
import os


def write_category_subset(input_path: str, output_path: str, category: str, limit: int) -> int:
    rows = []
    with open(input_path, "r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            raise ValueError(f"No header found in {input_path}")
        if "categories" not in fieldnames:
            raise ValueError(f"Expected a 'categories' column in {input_path}")
        for row in reader:
            labels = {part.strip().lower() for part in (row.get("categories") or "").split(",")}
            if category.lower() in labels:
                rows.append(row)
            if len(rows) >= limit:
                break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create prompt subsets for GeoClean rebuttal experiments.")
    parser.add_argument("--input", default="data/i2p_benchmark.csv")
    parser.add_argument("--output_dir", default="data/rebuttal")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument(
        "--categories",
        nargs="+",
        default=["sexual", "violence", "self-harm", "illegal activity", "harassment"],
    )
    args = parser.parse_args()

    for category in args.categories:
        name = category.replace(" ", "_").replace("-", "_")
        output_path = os.path.join(args.output_dir, f"i2p_{name}_{args.limit}.csv")
        count = write_category_subset(args.input, output_path, category, args.limit)
        print(f"{category}: wrote {count} prompts -> {output_path}")


if __name__ == "__main__":
    main()
