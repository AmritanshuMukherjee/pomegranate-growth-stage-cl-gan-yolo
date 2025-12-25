import cv2
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, yolo_root, split="train", transform=None):
        """
        Args:
            yolo_root (Path or str): root directory containing images/ and labels/
            split (str): train / val / test
            transform: Albumentations transform (must include ToTensorV2)
        """
        self.yolo_root = Path(yolo_root)
        self.split = split
        self.transform = transform

        self.images_dir = self.yolo_root / "images" / split
        self.labels_dir = self.yolo_root / "labels" / split

        if not self.images_dir.exists():
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png")) +
            list(self.images_dir.glob("*.jpeg"))
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

        self.label_files = [
            self.labels_dir / f"{img.stem}.txt" for img in self.image_files
        ]

    def __len__(self):
        return len(self.image_files)

    def _load_labels(self, label_path):
        """
        YOLO label format:
        class x_center y_center width height
        """
        if not label_path.exists():
            return torch.zeros((0, 5), dtype=torch.float32)

        labels = []
        with open(label_path, "r") as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) != 5:
                    continue
                labels.append([float(x) for x in parts])

        if len(labels) == 0:
            return torch.zeros((0, 5), dtype=torch.float32)

        return torch.tensor(labels, dtype=torch.float32)

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        label_path = self.label_files[idx]

        image = cv2.imread(str(image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        targets = self._load_labels(label_path)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]  # already torch.Tensor

        return image, targets


def yolo_collate_fn(batch):
    """
    Custom collate function for YOLO
    """
    images = []
    targets = []

    for i, (img, tgt) in enumerate(batch):
        images.append(img)
        if tgt.numel() > 0:
            batch_index = torch.full((tgt.shape[0], 1), i)
            targets.append(torch.cat([batch_index, tgt], dim=1))

    images = torch.stack(images, dim=0)

    if len(targets) > 0:
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6))

    return images, targets
