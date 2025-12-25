import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, root, split, transform=None):
        self.root = root
        self.split = split
        self.transform = transform

        self.images_dir = os.path.join(root, "images", split)
        self.labels_dir = os.path.join(root, "labels", split)

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted([
            f for f in os.listdir(self.images_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        ])

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        label_path = os.path.join(
            self.labels_dir, img_name.replace(".jpg", ".txt").replace(".png", ".txt")
        )

        boxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    boxes.append([cls, x, y, w, h])

        boxes = np.array(boxes, dtype=np.float32)

        if self.transform:
            transformed = self.transform(image=image)
            image = transformed["image"]

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        targets = torch.zeros((len(boxes), 6))
        if len(boxes) > 0:
            targets[:, 1:] = torch.from_numpy(boxes)

        targets[:, 0] = idx  # batch index placeholder

        return image, targets


def yolo_collate_fn(batch):
    images = []
    targets = []

    for i, (img, tgt) in enumerate(batch):
        images.append(img)
        if tgt.numel() > 0:
            tgt[:, 0] = i
            targets.append(tgt)

    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0) if len(targets) > 0 else torch.zeros((0, 6))

    return images, targets
