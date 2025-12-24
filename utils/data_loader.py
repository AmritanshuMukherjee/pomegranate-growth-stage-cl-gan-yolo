import torch
import cv2
import numpy as np
from pathlib import Path


class YOLODataset(torch.utils.data.Dataset):
    def __init__(self, yolo_root, split="train", transform=None):
        self.yolo_root = Path(yolo_root)
        self.split = split
        self.transform = transform

        self.images_dir = self.yolo_root / "images" / split
        self.labels_dir = self.yolo_root / "labels" / split

        if not self.images_dir.exists():
            raise RuntimeError(f"Images directory not found: {self.images_dir}")

        self.image_files = sorted(
            [p for p in self.images_dir.iterdir() if p.suffix.lower() in [".jpg", ".png", ".jpeg"]]
        )

        if len(self.image_files) == 0:
            raise RuntimeError(f"No images found in {self.images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        label_path = self.labels_dir / f"{img_path.stem}.txt"

        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []

        if label_path.exists():
            with open(label_path) as f:
                for line in f:
                    cls, x, y, w, h = map(float, line.split())

                    # 🔥 FILTER INVALID BOXES EARLY
                    if w <= 0 or h <= 0:
                        continue

                    if not (0 <= x <= 1 and 0 <= y <= 1):
                        continue

                    boxes.append([x, y, w, h])
                    labels.append(int(cls))

        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        if self.transform:
            transformed = self.transform(
                image=image,
                bboxes=boxes,
                class_labels=labels,
            )
            image = transformed["image"]

            if len(transformed["bboxes"]) > 0:
                boxes = torch.tensor(transformed["bboxes"], dtype=torch.float32)
                labels = torch.tensor(transformed["class_labels"], dtype=torch.float32).unsqueeze(1)
                targets = torch.cat([labels, boxes], dim=1)
            else:
                targets = torch.zeros((0, 5), dtype=torch.float32)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0
            targets = torch.zeros((0, 5), dtype=torch.float32)

        return image, targets


def yolo_collate_fn(batch):
    images = []
    targets = []

    for i, (img, t) in enumerate(batch):
        images.append(img)
        if t.numel() > 0:
            batch_idx = torch.full((t.shape[0], 1), i)
            targets.append(torch.cat([batch_idx, t], dim=1))

    images = torch.stack(images)

    if len(targets):
        targets = torch.cat(targets, dim=0)
    else:
        targets = torch.zeros((0, 6))

    return images, targets
