import csv
import json
from pathlib import Path

from PIL import Image
import torch

from src.attention import (
    ViTAttentionExtractor,
    compute_attention_rollout,
    compute_last_layer_attention,
    save_attention_grid,
    save_attention_visuals,
)
from src.dataset import MRNetDataset, get_transforms
from src.model import build_model


VALID_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def list_image_paths(input_path):
    input_path = Path(input_path)
    if input_path.is_file():
        return [input_path]

    return sorted(
        path for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    )


def load_checkpoint_model(cfg, checkpoint_path, device):
    class_names = MRNetDataset.CLASSES
    model = build_model(cfg, sit_weights_path=None, use_timm_pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model, class_names


def load_external_image(image_path, img_size):
    image = Image.open(image_path).convert("RGB")
    transform = get_transforms(img_size=img_size, split="val")
    tensor = transform(image)
    return image, tensor


def predict_tensor(model, image_tensor, device):
    image_batch = image_tensor.unsqueeze(0).to(device)
    logits = model(image_batch)
    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(probs.argmax().item())
    return pred_idx, probs


def save_external_predictions(records, output_dir):
    output_dir = Path(output_dir)

    with open(output_dir / "external_predictions.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)

    if not records:
        return

    with open(output_dir / "external_predictions.csv", "w", newline="", encoding="utf-8") as f:
        fieldnames = list(records[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def load_validation_metrics(metrics_path):
    metrics_path = Path(metrics_path)
    if not metrics_path.exists():
        return None

    with open(metrics_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_external_comparison(records, validation_metrics):
    if not records:
        return {
            "num_external_images": 0,
            "predicted_class_distribution": {},
            "confidence_mean": None,
            "validation_reference": validation_metrics,
        }

    predicted_class_distribution = {}
    confidences = []
    for record in records:
        predicted_class_distribution[record["pred_label"]] = (
            predicted_class_distribution.get(record["pred_label"], 0) + 1
        )
        confidences.append(record["pred_confidence"])

    total = len(records)
    predicted_class_distribution = {
        label: {
            "count": count,
            "fraction": count / total,
        }
        for label, count in predicted_class_distribution.items()
    }

    return {
        "num_external_images": total,
        "predicted_class_distribution": predicted_class_distribution,
        "confidence_mean": sum(confidences) / total,
        "validation_reference": validation_metrics,
        "notes": [
            "External internet MRI images may differ from MRNet in contrast, cropping, scanner protocol, resolution, and annotation quality.",
            "A confidence drop or skewed class distribution on external images can indicate domain shift.",
        ],
    }


def save_external_comparison(summary, output_dir):
    output_dir = Path(output_dir)
    with open(output_dir / "external_comparison.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)


def generate_external_attention_artifacts(
    model,
    image_tensor,
    image_name,
    device,
    output_dir,
    map_type="rollout",
):
    image_batch = image_tensor.unsqueeze(0).to(device)
    with ViTAttentionExtractor(model) as extractor:
        logits = model(image_batch)

    probs = torch.softmax(logits, dim=1)[0].detach().cpu()
    pred_idx = int(probs.argmax().item())
    attention_maps = extractor.get_attention_maps()
    if map_type == "rollout":
        attn_map = compute_attention_rollout(attention_maps)[0]
    else:
        attn_map = compute_last_layer_attention(attention_maps)[0]

    save_attention_visuals(image_tensor, attn_map, Path(output_dir) / image_name)
    return pred_idx, probs


def save_external_attention_overview(output_dir):
    output_dir = Path(output_dir)
    overlays = sorted(output_dir.glob("*_overlay.png"))
    save_attention_grid(
        [str(path) for path in overlays],
        title="External MRI attention overlays",
        output_path=output_dir / "external_attention_overview.png",
    )
