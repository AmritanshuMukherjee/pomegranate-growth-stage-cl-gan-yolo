import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader

from utils.data_loader import YOLODataset, yolo_collate_fn
from utils.augmentations import get_train_transforms, get_val_transforms


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    # ---------------------------
    # Dataset paths
    # ---------------------------
    yolo_root = Path(dataset_cfg["yolo"]["root"])

    train_transforms = get_train_transforms(dataset_cfg)
    val_transforms = get_val_transforms(dataset_cfg)

    train_dataset = YOLODataset(
        yolo_root=yolo_root,
        split="train",
        transform=train_transforms
    )

    val_dataset = YOLODataset(
        yolo_root=yolo_root,
        split="val",
        transform=val_transforms
    )

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    # ---------------------------
    # DataLoader (Windows-safe)
    # ---------------------------
    dl_cfg = train_cfg["dataloader"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=0,              # IMPORTANT: Windows safe
        pin_memory=False,
        collate_fn=yolo_collate_fn
    )

    # ---------------------------
    # Optimizer
    # ---------------------------
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        momentum=train_cfg["training"]["momentum"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )

    epochs = train_cfg["training"]["epochs"]

    # ===========================
    # TRAIN LOOP
    # ===========================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            loss_dict = model(images, targets)

            # YOLO returns a dict of losses
            if isinstance(loss_dict, dict):
                loss = sum(loss_dict.values())
            else:
                loss = loss_dict

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)

        logging.info(
            f"Epoch [{epoch + 1}/{epochs}] | Train Loss: {avg_loss:.4f}"
        )

    logging.info("✅ YOLO Phase-1 Training Completed")
