import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, yolo_root, split, transform=None):
        self.images_dir = os.path.join(yolo_root, "images", split)
        self.labels_dir = os.path.join(yolo_root, "labels", split)
        self.transform = transform

        if not os.path.exists(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(
            self.labels_dir, img_name.rsplit(".", 1)[0] + ".txt"
        )

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        targets = []
        if os.path.exists(label_path):
            with open(label_path) as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    targets.append([cls, x, y, w, h])

        targets = torch.tensor(targets, dtype=torch.float32)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        return image, targets
