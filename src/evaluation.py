import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize
from tqdm import tqdm


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()

    all_labels = []
    all_preds = []
    all_probs = []

    for images, labels in tqdm(loader, leave=False, desc="predict"):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)

        all_labels.append(labels.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    labels = np.concatenate(all_labels)
    preds = np.concatenate(all_preds)
    probs = np.concatenate(all_probs)
    return labels, preds, probs


def compute_specificity_per_class(conf_mat):
    specificities = []
    total = conf_mat.sum()

    for class_idx in range(conf_mat.shape[0]):
        tp = conf_mat[class_idx, class_idx]
        fn = conf_mat[class_idx, :].sum() - tp
        fp = conf_mat[:, class_idx].sum() - tp
        tn = total - tp - fn - fp
        specificities.append(float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0)

    return specificities


def compute_metrics(labels, preds, probs, class_names):
    conf_mat = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    sensitivities = conf_mat.diagonal() / np.maximum(conf_mat.sum(axis=1), 1)
    specificities = compute_specificity_per_class(conf_mat)

    labels_ovr = label_binarize(labels, classes=list(range(len(class_names))))
    roc_auc_per_class = {}
    for class_idx, class_name in enumerate(class_names):
        try:
            roc_auc_per_class[class_name] = float(
                roc_auc_score(labels_ovr[:, class_idx], probs[:, class_idx])
            )
        except ValueError:
            roc_auc_per_class[class_name] = None

    macro_roc_auc = roc_auc_score(
        labels_ovr,
        probs,
        average="macro",
        multi_class="ovr",
    )

    metrics = {
        "top1_accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro")),
        "f1_weighted": float(f1_score(labels, preds, average="weighted")),
        "sensitivity_per_class": {
            class_name: float(sensitivities[idx])
            for idx, class_name in enumerate(class_names)
        },
        "specificity_per_class": {
            class_name: float(specificities[idx])
            for idx, class_name in enumerate(class_names)
        },
        "roc_auc_per_class": roc_auc_per_class,
        "roc_auc_macro_ovr": float(macro_roc_auc),
        "confusion_matrix": conf_mat.tolist(),
    }
    return metrics, conf_mat


def save_confusion_matrix(conf_mat, class_names, output_path):
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        conf_mat,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
    )
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.title("Validation Confusion Matrix")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_roc_curves(labels, probs, class_names, output_path):
    labels_ovr = label_binarize(labels, classes=list(range(len(class_names))))

    plt.figure(figsize=(7, 6))
    for class_idx, class_name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(labels_ovr[:, class_idx], probs[:, class_idx])
            auc_score = roc_auc_score(labels_ovr[:, class_idx], probs[:, class_idx])
        except ValueError:
            continue

        plt.plot(fpr, tpr, label=f"{class_name} (AUC={auc_score:.3f})")

    plt.plot([0, 1], [0, 1], linestyle="--", color="grey")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("One-vs-Rest ROC Curves")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_prediction_table(labels, preds, probs, class_names, output_path):
    output_path = Path(output_path)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["sample_index", "true_label", "pred_label"] + [
            f"prob_{name}" for name in class_names
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for idx, (label, pred, prob_row) in enumerate(zip(labels, preds, probs)):
            row = {
                "sample_index": idx,
                "true_label": class_names[int(label)],
                "pred_label": class_names[int(pred)],
            }
            for class_name, prob in zip(class_names, prob_row):
                row[f"prob_{class_name}"] = float(prob)
            writer.writerow(row)


def evaluate_and_save(model, loader, device, class_names, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    labels, preds, probs = collect_predictions(model, loader, device)
    metrics, conf_mat = compute_metrics(labels, preds, probs, class_names)

    save_confusion_matrix(conf_mat, class_names, output_dir / "confusion_matrix.png")
    save_roc_curves(labels, probs, class_names, output_dir / "roc_curves.png")
    save_prediction_table(
        labels,
        preds,
        probs,
        class_names,
        output_dir / "validation_predictions.csv",
    )

    with open(output_dir / "evaluation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics
