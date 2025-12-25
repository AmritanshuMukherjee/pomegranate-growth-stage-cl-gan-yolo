import os
import torch
import logging
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.optim import SGD
from torch.utils.data import DataLoader

from utils.data_loader import YOLODataset
from utils.augmentations import get_train_transforms, get_val_transforms


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    # ==========================
    # Dataset path (Kaggle-safe)
    # ==========================
    yolo_root = dataset_cfg["yolo"]["root"]

    logging.info(f"Using Kaggle dataset: {yolo_root}")

    train_dataset = YOLODataset(yolo_root, "train", get_train_transforms(dataset_cfg))
    val_dataset = YOLODataset(yolo_root, "val", get_val_transforms(dataset_cfg))

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn
    )

    optimizer = SGD(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        momentum=0.937,
        weight_decay=5e-4
    )

    scaler = GradScaler()

    # ==========================
    # Create experiment folders
    # ==========================
    save_dir = "experiments/yolo_baseline"
    os.makedirs(save_dir, exist_ok=True)

    best_loss = float("inf")
    epochs = train_cfg["epochs"]

    # ==========================
    # TRAIN
    # ==========================
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]")

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
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # ==========================
        # SAVE BEST MODEL
        # ==========================
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": best_loss,
                },
                f"{save_dir}/best_model.pth"
            )
            logging.info("💾 Saved new best model")

    # ==========================
    # SAVE FINAL MODEL
    # ==========================
    torch.save(
        model.state_dict(),
        f"{save_dir}/final_model.pth"
    )

    logging.info("✅ YOLO Phase-1 training completed")
