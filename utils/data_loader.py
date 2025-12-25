import torch
from torch.utils.data import Dataset
from pathlib import Path
import cv2
import numpy as np


class YOLODataset(Dataset):
    def __init__(self, yolo_root: Path, split: str, transform=None):
        self.images_dir = yolo_root / "images" / split
        self.labels_dir = yolo_root / "labels" / split
        self.transform = transform

        if not self.images_dir.exists():
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted(
            list(self.images_dir.glob("*.jpg")) +
            list(self.images_dir.glob("*.png"))
        )

        if len(self.image_files) == 0:
            raise RuntimeError("No images found")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        targets = []
        if label_path.exists():
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.split())
                    targets.append([cls, x, y, w, h])

        targets = np.array(targets, dtype=np.float32)

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=targets[:, 1:] if len(targets) else [],
                class_labels=targets[:, 0].tolist() if len(targets) else [],
            )
            image = transformed["image"]
            if len(transformed["bboxes"]) > 0:
                targets = np.column_stack([
                    transformed["class_labels"],
                    transformed["bboxes"]
                ])
            else:
                targets = np.zeros((0, 5), dtype=np.float32)

        return image, torch.tensor(targets, dtype=torch.float32)

    @staticmethod
    def collate_fn(batch):
        images, targets = zip(*batch)
        images = torch.stack(images)
        max_targets = max(t.shape[0] for t in targets)

        padded_targets = []
        for t in targets:
            pad = torch.zeros((max_targets - t.shape[0], 5))
            padded_targets.append(torch.cat([t, pad], dim=0))

        return images, torch.stack(padded_targets)
