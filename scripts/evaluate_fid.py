import argparse
import csv
import os
from pathlib import Path

from cleanfid import fid


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute CleanFID against a reference image directory.")
    parser.add_argument("--reference", required=True)
    parser.add_argument("--outputs", nargs="+", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", default="clean")
    parser.add_argument("--output_csv", default=None)
    args = parser.parse_args()

    rows = []
    for output_dir in args.outputs:
        score = fid.compute_fid(args.reference, output_dir, mode=args.mode, device=args.device)
        rows.append(
            {
                "experiment": Path(output_dir).name,
                "reference": args.reference,
                "output_dir": output_dir,
                "fid": score,
            }
        )
        print(f"{Path(output_dir).name},fid,{score:.6f}")

    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        with open(args.output_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["experiment", "reference", "output_dir", "fid"])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
