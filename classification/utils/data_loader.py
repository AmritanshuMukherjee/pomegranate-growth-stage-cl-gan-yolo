import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset


class YOLODataset(Dataset):
    def __init__(self, root, split, img_size=640, transform=None):
        self.root = root
        self.split = split
        self.img_size = img_size
        self.transform = transform

        self.images_dir = os.path.join(root, "images", split)
        self.labels_dir = os.path.join(root, "labels", split)

        if not os.path.isdir(self.images_dir):
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted(
            f for f in os.listdir(self.images_dir)
            if f.endswith((".jpg", ".jpeg", ".png"))
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)

        # ---------------------------
        # Load image
        # ---------------------------
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        h0, w0 = image.shape[:2]

        # Resize (YOLO standard)
        image = cv2.resize(image, (self.img_size, self.img_size))

        # ---------------------------
        # Load labels
        # ---------------------------
        label_path = os.path.join(
            self.labels_dir,
            img_name.rsplit(".", 1)[0] + ".txt"
        )

        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f.readlines():
                    cls, x, y, w, h = map(float, line.strip().split())
                    targets.append([cls, x, y, w, h])

        targets = np.array(targets, dtype=np.float32)

        # ---------------------------
        # To tensor
        # ---------------------------
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # YOLO target format: [image_idx, class, x, y, w, h]
        target_tensor = torch.zeros((len(targets), 6))
        if len(targets) > 0:
            target_tensor[:, 1:] = torch.from_numpy(targets)

        target_tensor[:, 0] = idx

        return image, target_tensor


def yolo_collate_fn(batch):
    images, targets = [], []

    for i, (img, tgt) in enumerate(batch):
        images.append(img)
        if tgt.numel() > 0:
            tgt[:, 0] = i
            targets.append(tgt)

    images = torch.stack(images, 0)
    targets = torch.cat(targets, 0) if len(targets) else torch.zeros((0, 6))

    return images, targets
