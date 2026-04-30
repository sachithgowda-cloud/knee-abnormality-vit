import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms


class MRNetDataset(Dataset):
    """
    Loads MRNet .npy volumes and extracts 2D slices for classification.

    MRNet label priority for 3-class formulation:
        ACL tear  (acl=1)              → class 1
        Meniscal tear (meniscus=1)     → class 2
        Normal    (neither)            → class 0
    When both ACL and meniscus are positive, ACL takes priority.
    """

    CLASSES = ["normal", "acl", "meniscus"]

    def __init__(self, root_dir, split="train", plane="sagittal",
                 slices_per_volume=2, transform=None):
        self.root_dir = Path(root_dir)
        self.split = split
        self.plane = plane
        self.slices_per_volume = slices_per_volume
        self.transform = transform
        self.classes = self.CLASSES

        labels_df = self._load_labels()
        self.samples = self._build_samples(labels_df)

    def _load_labels(self):
        def read_csv(task):
            path = self.root_dir / f"{self.split}-{task}.csv"
            return pd.read_csv(path, header=None, names=["case", task])

        acl = read_csv("acl")
        men = read_csv("meniscus")
        df = acl.merge(men, on="case")

        def to_class(row):
            if row["acl"] == 1:
                return 1
            if row["meniscus"] == 1:
                return 2
            return 0

        df["label"] = df.apply(to_class, axis=1)
        return df

    def _build_samples(self, labels_df):
        vol_dir = self.root_dir / self.split / self.plane
        samples = []
        for _, row in labels_df.iterrows():
            npy_path = vol_dir / f"{int(row['case']):04d}.npy"
            if npy_path.exists():
                samples.append((npy_path, int(row["label"])))
        return samples

    def _extract_slice_indices(self, volume):
        """Return `slices_per_volume` representative slice indices."""
        n = volume.shape[0]
        if self.slices_per_volume == 1:
            return [n // 2]

        mid = n // 2
        # Slice with the highest total signal (most informative)
        signal_idx = int(np.argmax(volume.sum(axis=(1, 2))))
        if signal_idx == mid:
            signal_idx = min(mid + 1, n - 1)
        return [mid, signal_idx]

    def __len__(self):
        return len(self.samples) * self.slices_per_volume

    def get_sample_metadata(self, idx):
        vol_idx = idx // self.slices_per_volume
        slice_order = idx % self.slices_per_volume

        npy_path, label = self.samples[vol_idx]
        volume = np.load(npy_path)
        slice_indices = self._extract_slice_indices(volume)

        return {
            "sample_index": idx,
            "case_id": npy_path.stem,
            "file_path": str(npy_path),
            "label": int(label),
            "slice_index": int(slice_indices[slice_order]),
            "plane": self.plane,
            "split": self.split,
        }

    def __getitem__(self, idx):
        vol_idx = idx // self.slices_per_volume
        slice_order = idx % self.slices_per_volume

        npy_path, label = self.samples[vol_idx]
        volume = np.load(npy_path)          # shape: [N_slices, H, W]

        slice_indices = self._extract_slice_indices(volume)
        raw = volume[slice_indices[slice_order]]   # [H, W]

        # Normalise to [0, 255]
        raw = raw.astype(np.float32)
        lo, hi = raw.min(), raw.max()
        raw = (raw - lo) / (hi - lo + 1e-8)
        img = Image.fromarray((raw * 255).astype(np.uint8)).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, label


def get_transforms(img_size=224, split="train"):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    if split == "train":
        return transforms.Compose([
            transforms.Resize((img_size, img_size)),
            # Random X/Y translations (±10%), rotations ±20°, scaling 0.85-1.15
            transforms.RandomAffine(
                degrees=20,
                translate=(0.10, 0.10),
                scale=(0.85, 1.15),
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


def get_dataloaders(data_root, cfg):
    img_size  = cfg["model"]["img_size"]
    batch     = cfg["training"]["batch_size"]
    workers   = cfg["data"]["num_workers"]
    plane     = cfg["data"]["plane"]
    n_slices  = cfg["data"]["slices_per_volume"]
    tuning_ratio = cfg["data"].get("tuning_split_ratio", 0.0)
    seed = cfg["training"].get("seed", 42)

    split_map = {"train": "train", "val": "valid"}   # MRNet uses "valid" folder

    datasets = {
        name: MRNetDataset(
            root_dir=data_root,
            split=folder,
            plane=plane,
            slices_per_volume=n_slices,
            transform=get_transforms(img_size, name),
        )
        for name, folder in split_map.items()
    }

    if tuning_ratio > 0:
        train_dataset = datasets["train"]
        total_indices = np.arange(len(train_dataset))
        rng = np.random.default_rng(seed)
        rng.shuffle(total_indices)

        tune_size = int(len(total_indices) * tuning_ratio)
        if tune_size <= 0 or tune_size >= len(total_indices):
            raise ValueError(
                f"Invalid tuning_split_ratio={tuning_ratio}. It must create a non-empty split."
            )

        tune_indices = total_indices[:tune_size].tolist()
        train_indices = total_indices[tune_size:].tolist()
        datasets["train"] = Subset(train_dataset, train_indices)
        datasets["tune"] = Subset(train_dataset, tune_indices)

    loaders = {
        name: DataLoader(
            ds,
            batch_size=batch,
            shuffle=(name in {"train", "tune"}),
            num_workers=workers,
            pin_memory=True,
        )
        for name, ds in datasets.items()
    }

    return loaders, MRNetDataset.CLASSES


def get_dataset_labels(dataset):
    if isinstance(dataset, Subset):
        parent_labels = get_dataset_labels(dataset.dataset)
        return [parent_labels[idx] for idx in dataset.indices]

    if isinstance(dataset, MRNetDataset):
        labels = []
        for sample_idx in range(len(dataset)):
            vol_idx = sample_idx // dataset.slices_per_volume
            _, label = dataset.samples[vol_idx]
            labels.append(int(label))
        return labels

    raise TypeError(f"Unsupported dataset type for label extraction: {type(dataset)!r}")
