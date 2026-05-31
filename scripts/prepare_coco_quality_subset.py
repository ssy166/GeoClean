import argparse
import csv
import json
import os
import random
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urljoin

import requests


ANNOTATION_URL = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
VAL2014_IMAGE_ROOT = "http://images.cocodataset.org/val2014/"


def download_file(url: str, path: Path, retries: int = 5) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                with tmp_path.open("wb") as handle:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            handle.write(chunk)
            tmp_path.replace(path)
            return
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            if attempt == retries:
                raise
            time.sleep(2 * attempt)


def ensure_annotations(root: Path) -> Path:
    zip_path = root / "annotations_trainval2014.zip"
    captions_path = root / "annotations" / "captions_val2014.json"
    if not captions_path.exists():
        print(f"Downloading COCO annotations: {ANNOTATION_URL}")
        download_file(ANNOTATION_URL, zip_path)
        print(f"Extracting {captions_path.name}")
        with zipfile.ZipFile(zip_path) as archive:
            archive.extract("annotations/captions_val2014.json", root)
    return captions_path


def build_subset(captions_path: Path, sample_size: int, seed: int) -> list[dict]:
    with captions_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    image_by_id = {item["id"]: item for item in data["images"]}
    captions_by_image = {}
    for annotation in data["annotations"]:
        captions_by_image.setdefault(annotation["image_id"], []).append(annotation["caption"].strip())

    image_ids = sorted(captions_by_image)
    rng = random.Random(seed)
    selected_ids = rng.sample(image_ids, min(sample_size, len(image_ids)))

    rows = []
    for image_id in selected_ids:
        image = image_by_id[image_id]
        captions = sorted(captions_by_image[image_id])
        prompt = captions[0]
        file_name = image["file_name"]
        rows.append(
            {
                "image_id": image_id,
                "file_name": file_name,
                "prompt": prompt,
                "caption": prompt,
                "coco_url": urljoin(VAL2014_IMAGE_ROOT, file_name),
            }
        )
    return rows


def write_csv(rows: list[dict], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "file_name", "prompt", "caption", "coco_url", "reference_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def download_references(rows: list[dict], image_dir: Path, workers: int) -> None:
    image_dir.mkdir(parents=True, exist_ok=True)

    def fetch(row: dict) -> tuple[str, str]:
        path = image_dir / row["file_name"]
        download_file(row["coco_url"], path)
        return row["file_name"], str(path)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(fetch, row) for row in rows]
        completed = 0
        for future in as_completed(futures):
            file_name, path = future.result()
            completed += 1
            if completed % 50 == 0 or completed == len(rows):
                print(f"Downloaded {completed}/{len(rows)} reference images")
            for row in rows:
                if row["file_name"] == file_name:
                    row["reference_path"] = path
                    break


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a 500-sample COCO validation subset for T2I quality checks.")
    parser.add_argument("--root", default="data/mscoco_val2014_quality")
    parser.add_argument("--output_csv", default="data/mscoco_val2014_quality_500.csv")
    parser.add_argument("--sample_size", type=int, default=500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download_images", action="store_true")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.root)
    captions_path = ensure_annotations(root)
    rows = build_subset(captions_path, args.sample_size, args.seed)
    for row in rows:
        row["reference_path"] = str(root / "images" / row["file_name"])
    if args.download_images:
        download_references(rows, root / "images", args.workers)
    write_csv(rows, Path(args.output_csv))

    alias_path = Path("data/mscocoval2024_quality_500.csv")
    if Path(args.output_csv) != alias_path:
        write_csv(rows, alias_path)
    print(f"Wrote {len(rows)} rows -> {args.output_csv}")
    print(f"Wrote alias -> {alias_path}")


if __name__ == "__main__":
    main()
