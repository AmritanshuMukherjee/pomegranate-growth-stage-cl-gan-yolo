import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.data_loader import YOLODataset, yolo_collate_fn
from utils.augmentations import get_train_transforms, get_val_transforms


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    # =========================
    # DATASET PATH (KAGGLE SAFE)
    # =========================
    if "kaggle_root" in dataset_cfg:
        yolo_root = dataset_cfg["kaggle_root"]
        logging.info(f"Using Kaggle dataset: {yolo_root}")
    else:
        yolo_root = dataset_cfg["yolo"]["root"]

    train_transforms = get_train_transforms(dataset_cfg)
    val_transforms = get_val_transforms(dataset_cfg)

    train_dataset = YOLODataset(yolo_root, "train", train_transforms)
    val_dataset = YOLODataset(yolo_root, "val", val_transforms)

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    # =========================
    # DATALOADER
    # =========================
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=yolo_collate_fn,
    )

    optimizer = SGD(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        momentum=train_cfg["training"]["momentum"],
        weight_decay=train_cfg["training"]["weight_decay"],
    )

    scaler = GradScaler()
    epochs = train_cfg["training"]["epochs"]

    # =========================
    # TRAINING LOOP
    # =========================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

        for images, targets in pbar:
            images = images.to(device)

            # 🔥 CRITICAL FIX: ensure targets is Tensor
            if isinstance(targets, list):
                targets = torch.cat(targets, dim=0) if len(targets) > 0 else torch.zeros((0, 6))

            targets = targets.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss_dict = model(images, targets)

                if isinstance(loss_dict, dict):
                    loss = loss_dict["total_loss"]
                else:
                    loss = loss_dict

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

    logging.info("✅ YOLO Phase-1 training completed")
