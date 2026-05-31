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

    grouped = defaultdict(list)
    image_entries = []
    for start in tqdm(range(0, len(records), batch_size), desc="vision-text score"):
        batch = records[start : start + batch_size]
        images = [Image.open(record["path"]).convert("RGB") for record in batch]
        prompts = [record.get("prompt", "") for record in batch]
        inputs = processor(
            text=prompts,
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
        scores = (image_features * text_features).sum(dim=-1)
        image_features_cpu = image_features.detach().cpu()
        for record, score, image_feature in zip(batch, scores.detach().cpu().tolist(), image_features_cpu):
            grouped[record["method"]].append(float(score))
            image_entries.append(
                {
                    "method": record["method"],
                    "prompt_index": record.get("prompt_index"),
                    "image_feature": image_feature,
                }
            )

    base_features = {}
    for entry in image_entries:
        if entry["method"] == "base":
            base_features[entry["prompt_index"]] = entry["image_feature"]

    image_to_base = defaultdict(list)
    for entry in image_entries:
        base_feature = base_features.get(entry["prompt_index"])
        if base_feature is None:
            continue
        score = torch.dot(entry["image_feature"], base_feature).item()
        image_to_base[entry["method"]].append(float(score))

    summary = {}
    for method, scores in sorted(grouped.items()):
        tensor = torch.tensor(scores, dtype=torch.float32)
        base_tensor = torch.tensor(image_to_base.get(method, []), dtype=torch.float32)
        summary[method] = {
            "images": len(scores),
            "vision_text_cosine_mean": float(tensor.mean().item()),
            "vision_text_cosine_median": float(tensor.median().item()),
            "image_to_base_cosine_mean": float(base_tensor.mean().item()) if len(base_tensor) else None,
            "image_to_base_cosine_median": float(base_tensor.median().item()) if len(base_tensor) else None,
        }
    return summary


def write_outputs(input_dir: str, summary: dict) -> None:
    json_path = os.path.join(input_dir, "vision_text_score_summary.json")
    csv_path = os.path.join(input_dir, "vision_text_score_summary.csv")
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "method",
            "images",
            "vision_text_cosine_mean",
            "vision_text_cosine_median",
            "image_to_base_cosine_mean",
            "image_to_base_cosine_median",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for method, values in summary.items():
            writer.writerow({"method": method, **values})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute vision-text cosine scores for generated samples.")
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
        "method,images,vision_text_cosine_mean,vision_text_cosine_median,"
        "image_to_base_cosine_mean,image_to_base_cosine_median"
    )
    for method, values in summary.items():
        base_mean = values["image_to_base_cosine_mean"]
        base_median = values["image_to_base_cosine_median"]
        base_mean_text = f"{base_mean:.6f}" if base_mean is not None else ""
        base_median_text = f"{base_median:.6f}" if base_median is not None else ""
        print(
            f"{method},{values['images']},"
            f"{values['vision_text_cosine_mean']:.6f},{values['vision_text_cosine_median']:.6f},"
            f"{base_mean_text},{base_median_text}"
        )


if __name__ == "__main__":
    main()
