import os
import time
import torch
import logging
from pathlib import Path
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.data_loader import YOLODataset
from utils.augmentations import get_train_transforms, get_val_transforms


def resolve_yolo_root():
    kaggle_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    if os.path.exists(kaggle_root):
        logging.info(f"Using Kaggle dataset: {kaggle_root}")
        return kaggle_root
    return "data/yolo"


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    yolo_root = resolve_yolo_root()

    train_tfms = get_train_transforms(dataset_cfg)
    val_tfms = get_val_transforms(dataset_cfg)

    train_ds = YOLODataset(yolo_root, "train", train_tfms)
    val_ds = YOLODataset(yolo_root, "val", val_tfms)

    logging.info(f"Train samples: {len(train_ds)}")
    logging.info(f"Val samples: {len(val_ds)}")

    dl_cfg = train_cfg["dataloader"]

    train_loader = DataLoader(
        train_ds,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        momentum=train_cfg["training"]["momentum"],
        weight_decay=train_cfg["training"]["weight_decay"]
    )

    scaler = GradScaler()
    epochs = train_cfg["training"]["epochs"]

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, targets in pbar:
            images = torch.stack(images).to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()

            with autocast():
                loss_dict = model(images, targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        logging.info(
            f"Epoch [{epoch+1}/{epochs}] | Train Loss: {epoch_loss/len(train_loader):.4f}"
        )
