import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


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


def train(model, loaders, cfg, output_dir, device, optimizer):
    epochs   = cfg["training"]["epochs"]
    patience = cfg["training"]["early_stopping_patience"]

    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()
    writer    = SummaryWriter(log_dir=str(output_dir / "tensorboard"))

    best_val_acc   = 0.0
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
            no_improve   = 0
            torch.save(model.state_dict(), output_dir / "best_model.pth")
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}")
                break

    writer.close()
    print(f"\nBest val accuracy: {best_val_acc:.4f}")
    return history
