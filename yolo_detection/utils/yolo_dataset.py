import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np


class YOLODetectionDataset(Dataset):
    def __init__(self, root, split="train", img_size=640):
        self.img_dir = os.path.join(root, "images", split)
        self.lbl_dir = os.path.join(root, "labels", split)
        self.img_size = img_size

        self.images = sorted([
            f for f in os.listdir(self.img_dir)
            if f.endswith((".jpg", ".png"))
        ])

        assert len(self.images) > 0, f"No images found in {self.img_dir}"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]

        img_path = os.path.join(self.img_dir, img_name)
        lbl_path = os.path.join(self.lbl_dir, img_name.replace(".jpg", ".txt"))

        # Load image
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        # Load labels
        targets = []
        if os.path.exists(lbl_path):
            with open(lbl_path) as f:
                for line in f.readlines():
                    cls, xc, yc, w, h = map(float, line.split())
                    targets.append([cls, xc, yc, w, h])

        targets = torch.tensor(targets) if len(targets) else torch.zeros((0, 5))

        return img, targets
