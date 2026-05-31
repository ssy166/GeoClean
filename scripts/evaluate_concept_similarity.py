import argparse
import csv
import json
import os
from collections import defaultdict

import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


def load_records(input_dir: str) -> list[dict]:
    metadata_path = os.path.join(input_dir, "metadata.jsonl")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"metadata.jsonl not found in {input_dir}")

    records = []
    with open(metadata_path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            path = record.get("path")
            if path and not os.path.isabs(path):
                path = os.path.join(os.getcwd(), path)
            if path and os.path.exists(path):
                record["path"] = path
                records.append(record)
    return records


def feature_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if hasattr(value, "pooler_output") and value.pooler_output is not None:
        return value.pooler_output
    if hasattr(value, "last_hidden_state"):
        return value.last_hidden_state[:, 0]
    return value[0]


@torch.no_grad()
def score_records(records: list[dict], model_name: str, device: str, batch_size: int) -> dict:
    model = AutoModel.from_pretrained(model_name, use_safetensors=True).to(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model.eval()

    grouped_scores = defaultdict(list)
    paired_entries = []
    for start in tqdm(range(0, len(records), batch_size), desc="concept similarity"):
        batch = records[start : start + batch_size]
        images = [Image.open(record["path"]).convert("RGB") for record in batch]
        concepts = [record.get("concept", "") for record in batch]
        inputs = processor(
            text=concepts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        image_features = feature_tensor(model.get_image_features(pixel_values=inputs["pixel_values"]))
        text_kwargs = {"input_ids": inputs["input_ids"]}
        if "attention_mask" in inputs:
            text_kwargs["attention_mask"] = inputs["attention_mask"]
        text_features = feature_tensor(model.get_text_features(**text_kwargs))
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        scores = (image_features * text_features).sum(dim=-1).detach().cpu().tolist()

        for record, score in zip(batch, scores):
            method = record["method"]
            score = float(score)
            grouped_scores[method].append(score)
            paired_entries.append(
                {
                    "method": method,
                    "prompt_index": record.get("prompt_index"),
                    "concept_score": score,
                }
            )

    base_scores = {}
    for entry in paired_entries:
        if entry["method"] == "base":
            base_scores[entry["prompt_index"]] = entry["concept_score"]

    paired_reductions = defaultdict(list)
    for entry in paired_entries:
        base_score = base_scores.get(entry["prompt_index"])
        if base_score is None:
            continue
        paired_reductions[entry["method"]].append(base_score - entry["concept_score"])

    summary = {}
    for method, scores in sorted(grouped_scores.items()):
        score_tensor = torch.tensor(scores, dtype=torch.float32)
        reduction_tensor = torch.tensor(paired_reductions.get(method, []), dtype=torch.float32)
        summary[method] = {
            "images": len(scores),
            "concept_cosine_mean": float(score_tensor.mean().item()),
            "concept_cosine_median": float(score_tensor.median().item()),
            "paired_reduction_vs_base_mean": (
                float(reduction_tensor.mean().item()) if len(reduction_tensor) else None
            ),
            "paired_reduction_vs_base_median": (
                float(reduction_tensor.median().item()) if len(reduction_tensor) else None
            ),
        }
    return summary


def write_outputs(input_dir: str, summary: dict) -> None:
    json_path = os.path.join(input_dir, "concept_score_summary.json")
    csv_path = os.path.join(input_dir, "concept_score_summary.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "images",
            "concept_cosine_mean",
            "concept_cosine_median",
            "paired_reduction_vs_base_mean",
            "paired_reduction_vs_base_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, values in summary.items():
            writer.writerow({"method": method, **values})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute image-to-erased-concept similarity for generated samples.")
    parser.add_argument("--input", required=True, help="Directory containing metadata.jsonl and images.")
    parser.add_argument("--model_name", default="google/siglip-base-patch16-224")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch_size", type=int, default=16)
    args = parser.parse_args()

    records = load_records(args.input)
    print(f"Loaded {len(records)} records from {args.input}")
    summary = score_records(records, args.model_name, args.device, args.batch_size)
    write_outputs(args.input, summary)

    print(f"Scoring model: {args.model_name}")
    print(
        "method,images,concept_cosine_mean,concept_cosine_median,"
        "paired_reduction_vs_base_mean,paired_reduction_vs_base_median"
    )
    for method, values in summary.items():
        mean_reduction = values["paired_reduction_vs_base_mean"]
        median_reduction = values["paired_reduction_vs_base_median"]
        mean_reduction_text = f"{mean_reduction:.6f}" if mean_reduction is not None else ""
        median_reduction_text = f"{median_reduction:.6f}" if median_reduction is not None else ""
        print(
            f"{method},{values['images']},"
            f"{values['concept_cosine_mean']:.6f},{values['concept_cosine_median']:.6f},"
            f"{mean_reduction_text},{median_reduction_text}"
        )


if __name__ == "__main__":
    main()
