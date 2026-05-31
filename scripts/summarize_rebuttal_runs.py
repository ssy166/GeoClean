import argparse
import json
import os
from collections import defaultdict
from statistics import mean, median


def read_records(path: str):
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize GeoClean rebuttal generation metadata.")
    parser.add_argument("--root", default="results/rebuttal_flux")
    args = parser.parse_args()

    rows = []
    for dirpath, _, filenames in os.walk(args.root):
        if "metadata.jsonl" not in filenames:
            continue
        experiment = os.path.relpath(dirpath, args.root)
        grouped = defaultdict(list)
        for record in read_records(os.path.join(dirpath, "metadata.jsonl")):
            grouped[record.get("method", "unknown")].append(record)
        for method, records in sorted(grouped.items()):
            latencies = [float(r["latency_sec"]) for r in records if "latency_sec" in r]
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "count": len(records),
                    "avg_latency_sec": mean(latencies) if latencies else None,
                    "median_latency_sec": median(latencies) if latencies else None,
                }
            )

    if not rows:
        print("No metadata.jsonl files found.")
        return

    print("experiment,method,count,avg_latency_sec,median_latency_sec")
    for row in rows:
        avg = "" if row["avg_latency_sec"] is None else f"{row['avg_latency_sec']:.4f}"
        med = "" if row["median_latency_sec"] is None else f"{row['median_latency_sec']:.4f}"
        print(f"{row['experiment']},{row['method']},{row['count']},{avg},{med}")


if __name__ == "__main__":
    main()
