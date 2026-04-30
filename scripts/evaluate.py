import argparse
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import MRNetDataset, get_dataloaders
from src.evaluation import evaluate_and_save
from src.model import build_model


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained ViT-Small knee MRI classifier."
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
        help="Directory where evaluation artifacts will be saved.",
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


def resolve_output_dir(args, checkpoint_path):
    if args.output_dir:
        return Path(args.output_dir)
    return checkpoint_path.parent / "evaluation"


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = resolve_device(args)

    data_root = resolve_data_root(cfg, args)
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_dir = resolve_output_dir(args, checkpoint_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using device   : {device}")
    print(f"Dataset root   : {data_root}")
    print(f"Checkpoint     : {checkpoint_path}")
    print(f"Output dir     : {output_dir}")

    loaders, _ = get_dataloaders(data_root, cfg)
    class_names = MRNetDataset.CLASSES

    model = build_model(cfg, sit_weights_path=None, use_timm_pretrained=False)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model = model.to(device)

    metrics = evaluate_and_save(
        model=model,
        loader=loaders["val"],
        device=device,
        class_names=class_names,
        output_dir=output_dir,
    )

    print("\nEvaluation complete.")
    print(f"Top-1 accuracy : {metrics['top1_accuracy']:.4f}")
    print(f"Macro F1       : {metrics['f1_macro']:.4f}")
    print(f"Macro ROC-AUC  : {metrics['roc_auc_macro_ovr']:.4f}")
    print(f"Metrics file   : {output_dir / 'evaluation_metrics.json'}")
    print(f"Confusion plot : {output_dir / 'confusion_matrix.png'}")
    print(f"ROC plot       : {output_dir / 'roc_curves.png'}")


if __name__ == "__main__":
    main()
