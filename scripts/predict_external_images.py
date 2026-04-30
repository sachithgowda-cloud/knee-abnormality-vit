import argparse
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import (
    build_external_comparison,
    generate_external_attention_artifacts,
    list_image_paths,
    load_checkpoint_model,
    load_external_image,
    load_validation_metrics,
    predict_tensor,
    save_external_attention_overview,
    save_external_comparison,
    save_external_predictions,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the trained knee MRI classifier on external/public MRI images."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to one image or a folder of images.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where predictions and visualizations will be saved.",
    )
    parser.add_argument(
        "--validation-metrics",
        default=None,
        help="Optional evaluation_metrics.json for comparison against MRNet validation performance.",
    )
    parser.add_argument(
        "--with-attention",
        action="store_true",
        help="Also save attention heatmaps and overlays for external images.",
    )
    parser.add_argument(
        "--map-type",
        default="rollout",
        choices=["rollout", "last_layer"],
        help="Attention map type to generate when --with-attention is used.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to run inference on, e.g. cpu, cuda, cuda:0.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_output_dir(args, checkpoint_path):
    if args.output_dir:
        return Path(args.output_dir)
    return checkpoint_path.parent / "external_images"


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    image_paths = list_image_paths(args.input)
    if not image_paths:
        raise FileNotFoundError(f"No supported images found in: {args.input}")

    output_dir = resolve_output_dir(args, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, class_names = load_checkpoint_model(cfg, checkpoint_path, device)
    print(f"Using device   : {device}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Input images   : {len(image_paths)}")
    print(f"Output dir     : {output_dir}")

    records = []
    for image_path in image_paths:
        _, image_tensor = load_external_image(image_path, cfg["model"]["img_size"])

        if args.with_attention:
            pred_idx, probs = generate_external_attention_artifacts(
                model=model,
                image_tensor=image_tensor,
                image_name=image_path.stem,
                device=device,
                output_dir=output_dir,
                map_type=args.map_type,
            )
        else:
            pred_idx, probs = predict_tensor(model, image_tensor, device)

        records.append({
            "image_path": str(image_path),
            "pred_label": class_names[pred_idx],
            "pred_confidence": float(probs[pred_idx].item()),
            **{
                f"prob_{class_name}": float(probs[idx].item())
                for idx, class_name in enumerate(class_names)
            },
        })

    save_external_predictions(records, output_dir)

    validation_metrics = None
    if args.validation_metrics:
        validation_metrics = load_validation_metrics(args.validation_metrics)

    comparison = build_external_comparison(records, validation_metrics)
    save_external_comparison(comparison, output_dir)

    if args.with_attention:
        save_external_attention_overview(output_dir)

    print(f"\nPredictions    : {output_dir / 'external_predictions.json'}")
    print(f"Comparison     : {output_dir / 'external_comparison.json'}")
    if args.with_attention:
        print(f"Attention grid : {output_dir / 'external_attention_overview.png'}")


if __name__ == "__main__":
    main()
