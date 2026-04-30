import json

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import get_dataset_labels


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="train"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for images, labels in tqdm(loader, leave=False, desc="eval "):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(1) == labels).sum().item()
        total += images.size(0)
    return total_loss / total, correct / total


def build_criterion(train_dataset, cfg, device):
    loss_cfg = cfg["training"].get("loss", {})
    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0))
    use_class_weights = bool(loss_cfg.get("use_class_weights", True))

    weight_tensor = None
    if use_class_weights:
        labels = np.array(get_dataset_labels(train_dataset))
        class_counts = np.bincount(labels, minlength=cfg["model"]["num_classes"])
        class_weights = class_counts.sum() / np.maximum(class_counts, 1)
        class_weights = class_weights / class_weights.mean()
        weight_tensor = torch.tensor(class_weights, dtype=torch.float32, device=device)
        print(f"Class weights  : {class_weights.round(4).tolist()}")

    return nn.CrossEntropyLoss(
        weight=weight_tensor,
        label_smoothing=label_smoothing,
    )


def train(model, loaders, cfg, output_dir, device, optimizer):
    epochs   = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]
    warmup_epochs = min(cfg["training"].get("warmup_epochs", 0), max(epochs - 1, 0))

    cosine_epochs = max(epochs - warmup_epochs, 1)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs)
    if warmup_epochs > 0:
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        scheduler = cosine_scheduler

    criterion = build_criterion(loaders["train"].dataset, cfg, device)
    writer    = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    best_val_acc   = 0.0
    best_epoch     = 0
    no_improve     = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, loaders["train"], optimizer, criterion, device)
        val_loss,   val_acc   = evaluate(model, loaders["val"],   criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        writer.add_scalars("loss",     {"train": train_loss, "val": val_loss}, epoch)
        writer.add_scalars("accuracy", {"train": train_acc,  "val": val_acc},  epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss={train_loss:.4f} acc={train_acc:.4f} | "
            f"val   loss={val_loss:.4f}   acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch   = epoch
            no_improve   = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    torch.save(model.state_dict(), output_dir / "last_model.pth")

    metrics = {
        "best_val_accuracy": best_val_acc,
        "best_epoch": best_epoch,
        "epochs_completed": len(history["train_loss"]),
    }
    with open(output_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    writer.close()
    print(f"\nBest val accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
    return history, metrics
