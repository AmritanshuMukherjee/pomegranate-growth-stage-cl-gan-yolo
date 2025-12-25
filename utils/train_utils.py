import os
import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.data_loader import YOLODataset
from utils.augmentations import get_train_transforms, get_val_transforms


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    # ✅ Kaggle dataset path (verified)
    yolo_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    logging.info(f"Using Kaggle dataset: {yolo_root}")

    # =========================
    # DATA
    # =========================
    train_transforms = get_train_transforms(dataset_cfg)
    val_transforms = get_val_transforms(dataset_cfg)

    train_dataset = YOLODataset(yolo_root, "train", train_transforms)
    val_dataset = YOLODataset(yolo_root, "val", val_transforms)

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    dl_cfg = train_cfg["dataloader"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    # =========================
    # OPTIMIZER
    # =========================
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        momentum=train_cfg["training"]["momentum"],
        weight_decay=train_cfg["training"]["weight_decay"],
    )

    epochs = train_cfg["training"]["epochs"]
    scaler = GradScaler()

    # =========================
    # MODEL SAVE PATH
    # =========================
    exp_dir = Path("experiments/yolo_baseline")
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")

    # =========================
    # TRAIN LOOP
    # =========================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch+1}/{epochs}]",
            ncols=100,
        )

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with autocast():
                loss_dict = model(images, targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)

        logging.info(
            f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}"
        )

        # =========================
        # SAVE LAST MODEL
        # =========================
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "loss": avg_loss,
            },
            exp_dir / "last_model.pt",
        )

        # =========================
        # SAVE BEST MODEL
        # =========================
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                exp_dir / "best_model.pt",
            )
            logging.info(f"🔥 New best model saved (loss={best_loss:.4f})")

    logging.info("✅ YOLO Phase-1 training completed")
