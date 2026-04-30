import argparse
import json
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.model import build_model, get_optimizer
from src.trainer import train


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a ViT-Small knee MRI classifier on MRNet."
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
        "--output-dir",
        default=None,
        help="Override output directory. Defaults to a timestamped run folder.",
    )
    parser.add_argument(
        "--sit-weights",
        default=None,
        help="Optional path to SiT pretrained weights (.pth).",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on, e.g. cpu, cuda, cuda:0.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible training.",
    )
    parser.add_argument(
        "--colab",
        action="store_true",
        help="Use Colab paths from the config file.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_data_root(cfg, args):
    if args.data_root:
        return Path(args.data_root)
    key = "colab_root" if args.colab else "local_root"
    return Path(cfg["data"][key])


def resolve_output_dir(cfg, args):
    if args.output_dir:
        return Path(args.output_dir)

    base_key = "colab_output" if args.colab else "local_output"
    base_dir = Path(cfg["output"][base_key])
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    return base_dir / run_name


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def save_run_config(output_dir, cfg, args, data_root, device, sit_weights_path):
    run_config = {
        "config": cfg,
        "runtime": {
            "seed": args.seed,
            "colab": args.colab,
            "data_root": str(data_root),
            "device": str(device),
            "sit_weights": str(sit_weights_path) if sit_weights_path else None,
        },
    }
    with open(output_dir / "run_config.json", "w", encoding="utf-8") as f:
        json.dump(run_config, f, indent=2)


def main():
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(args.seed)
    device = resolve_device(args)
    data_root = resolve_data_root(cfg, args)
    output_dir = resolve_output_dir(cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)

    sit_weights_path = Path(args.sit_weights) if args.sit_weights else None
    if sit_weights_path is not None and not sit_weights_path.exists():
        raise FileNotFoundError(f"SiT weights not found: {sit_weights_path}")

    print(f"Using device   : {device}")
    print(f"Dataset root   : {data_root}")
    print(f"Output dir     : {output_dir}")
    if sit_weights_path:
        print(f"SiT weights    : {sit_weights_path}")
    else:
        print("SiT weights    : not provided, using timm pretrained weights")

    loaders, classes = get_dataloaders(data_root, cfg)
    print(f"Classes        : {classes}")
    print(f"Train samples  : {len(loaders['train'].dataset)}")
    print(f"Valid samples  : {len(loaders['val'].dataset)}")

    model = build_model(cfg, sit_weights_path=sit_weights_path).to(device)
    optimizer = get_optimizer(model, cfg)

    save_run_config(output_dir, cfg, args, data_root, device, sit_weights_path)
    _, metrics = train(model, loaders, cfg, output_dir, device, optimizer)

    print("\nTraining complete.")
    print(f"Best checkpoint: {output_dir / 'best_model.pth'}")
    print(f"Last checkpoint: {output_dir / 'last_model.pth'}")
    print(f"Metrics file   : {output_dir / 'metrics.json'}")
    print(f"History file   : {output_dir / 'history.json'}")
    print(f"Best val acc   : {metrics['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
