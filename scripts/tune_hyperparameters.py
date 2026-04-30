import argparse
import copy
import itertools
import json
from pathlib import Path
import sys

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
        description="Run a simple hyperparameter sweep for the ViT knee MRI classifier."
    )
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--sit-weights", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--colab", action="store_true")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[32])
    parser.add_argument("--epochs", nargs="+", type=int, default=[15, 25])
    parser.add_argument("--backbone-lrs", nargs="+", type=float, default=[1e-5, 5e-5])
    parser.add_argument("--head-lrs", nargs="+", type=float, default=[1e-4, 5e-4])
    parser.add_argument("--weight-decays", nargs="+", type=float, default=[1e-4, 1e-3])
    parser.add_argument("--tuning-split-ratio", type=float, default=0.15)
    return parser.parse_args()


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
    return Path(cfg["output"][base_key]) / "hyperparameter_sweeps"


def resolve_device(args):
    if args.device:
        return torch.device(args.device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_run_configs(base_cfg, args):
    combos = itertools.product(
        args.batch_sizes,
        args.epochs,
        args.backbone_lrs,
        args.head_lrs,
        args.weight_decays,
    )

    run_cfgs = []
    for batch_size, epochs, backbone_lr, head_lr, weight_decay in combos:
        cfg = copy.deepcopy(base_cfg)
        cfg["training"]["batch_size"] = batch_size
        cfg["training"]["epochs"] = epochs
        cfg["training"]["backbone_lr"] = backbone_lr
        cfg["training"]["head_lr"] = head_lr
        cfg["training"]["weight_decay"] = weight_decay
        cfg["data"]["tuning_split_ratio"] = args.tuning_split_ratio
        run_cfgs.append(cfg)
    return run_cfgs


def run_name(cfg):
    return (
        f"bs{cfg['training']['batch_size']}"
        f"_ep{cfg['training']['epochs']}"
        f"_blr{cfg['training']['backbone_lr']}"
        f"_hlr{cfg['training']['head_lr']}"
        f"_wd{cfg['training']['weight_decay']}"
    )


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def main():
    args = parse_args()
    base_cfg = load_config(args.config)
    data_root = resolve_data_root(base_cfg, args)
    output_dir = resolve_output_dir(base_cfg, args)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args)
    sit_weights = Path(args.sit_weights) if args.sit_weights else None

    all_results = []
    for cfg in build_run_configs(base_cfg, args):
        current_run_dir = output_dir / run_name(cfg)
        current_run_dir.mkdir(parents=True, exist_ok=True)

        loaders, _ = get_dataloaders(data_root, cfg)
        eval_loader = loaders.get("tune", loaders["val"])

        model = build_model(cfg, sit_weights_path=sit_weights).to(device)
        optimizer = get_optimizer(model, cfg)

        history, metrics = train(
            model=model,
            loaders={"train": loaders["train"], "val": eval_loader},
            cfg=cfg,
            output_dir=current_run_dir,
            device=device,
            optimizer=optimizer,
        )

        result = {
            "run_name": current_run_dir.name,
            "batch_size": cfg["training"]["batch_size"],
            "epochs": cfg["training"]["epochs"],
            "backbone_lr": cfg["training"]["backbone_lr"],
            "head_lr": cfg["training"]["head_lr"],
            "weight_decay": cfg["training"]["weight_decay"],
            "tuning_split_ratio": cfg["data"]["tuning_split_ratio"],
            "best_val_accuracy": metrics["best_val_accuracy"],
            "best_epoch": metrics["best_epoch"],
            "epochs_completed": metrics["epochs_completed"],
        }
        all_results.append(result)
        save_json(result, current_run_dir / "sweep_result.json")
        save_json(history, current_run_dir / "sweep_history.json")

    all_results.sort(key=lambda row: row["best_val_accuracy"], reverse=True)
    save_json(all_results, output_dir / "sweep_results.json")
    if all_results:
        save_json(all_results[0], output_dir / "best_sweep_result.json")

    print(f"Completed runs : {len(all_results)}")
    print(f"Results file   : {output_dir / 'sweep_results.json'}")
    if all_results:
        print(f"Best run       : {all_results[0]['run_name']}")
        print(f"Best accuracy  : {all_results[0]['best_val_accuracy']:.4f}")


if __name__ == "__main__":
    main()
