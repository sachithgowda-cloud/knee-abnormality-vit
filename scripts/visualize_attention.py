import argparse
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.attention import (
    ViTAttentionExtractor,
    compute_attention_rollout,
    compute_last_layer_attention,
    save_attention_grid,
    save_attention_visuals,
    summarize_attention_run,
)
from src.dataset import MRNetDataset, get_transforms
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate class attention maps and overlays for MRI slices."
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--data-root",
        default=None,
        help="Override dataset root directory.",
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Path to trained model checkpoint (.pth).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory where attention visualizations will be saved.",
    )
    parser.add_argument(
        "--split",
        default="valid",
        choices=["train", "valid"],
        help="Dataset split to visualize.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=12,
        help="Maximum number of samples to visualize.",
    )
    parser.add_argument(
        "--selection",
        default="correct",
        choices=["all", "correct", "incorrect"],
        help="Which predictions to visualize.",
    )
    parser.add_argument(
        "--map-type",
        default="rollout",
        choices=["rollout", "last_layer"],
        help="Attention map type to generate.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to evaluate on, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Use Colab paths from the config file.",
    )
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg, args):
    if args.data_root:
        return Path(args.data_root)
    key = "colab_root" if args.colab else "local_root"
    return Path(cfg["data"][key])


def resolve_output_dir(args, checkpoint_path, split, map_type):
    if args.output_dir:
        return Path(args.output_dir)
    return checkpoint_path.parent / f"attention_{split}_{map_type}"


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def should_keep_sample(selection, is_correct):
    if selection == "all":
        return True
    if selection == "correct":
        return is_correct
    return not is_correct


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args)

    data_root = resolve_data_root(cfg, args)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = resolve_output_dir(args, checkpoint_path, args.split, args.map_type)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MRNetDataset(
        root_dir=data_root,
        split=args.split,
        plane=cfg["data"]["plane"],
        slices_per_volume=cfg["data"]["slices_per_volume"],
        transform=get_transforms(cfg["model"]["img_size"], split="val"),
    )

    class_names = MRNetDataset.CLASSES
    model = build_model(cfg, sit_weights_path=None, use_timm_pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Using device   : {device}")
    print(f"Dataset root   : {data_root}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Output dir     : {output_dir}")
    print(f"Split          : {args.split}")
    print(f"Selection      : {args.selection}")
    print(f"Map type       : {args.map_type}")

    records = []
    overlay_paths = []

    for sample_idx in range(len(dataset)):
        image, label = dataset[sample_idx]
        image_batch = image.unsqueeze(0).to(device)

        with ViTAttentionExtractor(model) as extractor:
            logits = model(image_batch)

        probs = torch.softmax(logits, dim=1)
        pred = int(probs.argmax(dim=1).item())
        label = int(label)
        is_correct = pred == label

        if not should_keep_sample(args.selection, is_correct):
            continue

        attention_maps = extractor.get_attention_maps()
        if args.map_type == "rollout":
            attn_map = compute_attention_rollout(attention_maps)[0]
        else:
            attn_map = compute_last_layer_attention(attention_maps)[0]

        metadata = dataset.get_sample_metadata(sample_idx)
        stem = (
            f"sample_{sample_idx:04d}_case_{metadata['case_id']}"
            f"_slice_{metadata['slice_index']}"
            f"_{class_names[label]}_pred_{class_names[pred]}"
        )
        save_attention_visuals(image, attn_map, output_dir / stem)
        overlay_path = output_dir / f"{stem}_overlay.png"
        overlay_paths.append(str(overlay_path))

        records.append({
            "sample_index": sample_idx,
            "case_id": metadata["case_id"],
            "slice_index": metadata["slice_index"],
            "true_label": class_names[label],
            "pred_label": class_names[pred],
            "is_correct": bool(is_correct),
            "max_probability": float(probs[0, pred].item()),
            "map_type": args.map_type,
            "overlay_path": str(overlay_path),
            "image_path": str(output_dir / f"{stem}_image.png"),
            "attention_path": str(output_dir / f"{stem}_attention.png"),
        })

        if len(records) >= args.num_samples:
            break

    summarize_attention_run(records, output_dir)
    save_attention_grid(
        overlay_paths,
        title=f"Attention overlays ({args.selection}, {args.map_type})",
        output_path=output_dir / "attention_overview.png",
    )

    print(f"\nSaved {len(records)} attention visualizations.")
    print(f"Metadata JSON  : {output_dir / 'attention_metadata.json'}")
    print(f"Metadata CSV   : {output_dir / 'attention_metadata.csv'}")
    print(f"Overview image : {output_dir / 'attention_overview.png'}")


if __name__ == "__main__":
    main()
