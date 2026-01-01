import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class ClassificationDataset(Dataset):
    """
    Classification dataset built from YOLO labels.
    Uses ONLY class index (ignores bounding boxes).
    """

    def __init__(self, root, split="train", img_size=224):
        self.root = root
        self.split = split
        self.img_size = img_size

        self.images_dir = os.path.join(root, "images", split)
        self.labels_dir = os.path.join(root, "labels", split)

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted(
            f for f in os.listdir(self.images_dir)
            if f.endswith((".jpg", ".png", ".jpeg"))
        )

        if len(self.image_files) == 0:
            raise RuntimeError("No images found")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # ---- image ----
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # ---- label (class only) ----
        label_path = os.path.join(
            self.labels_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        cls = 0  # default fallback
        if os.path.exists(label_path):
            with open(label_path) as f:
                cls = int(float(f.readline().split()[0]))

        return image, cls
