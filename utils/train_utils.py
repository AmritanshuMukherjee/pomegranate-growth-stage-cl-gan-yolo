import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from utils.data_loader import YOLODataset, yolo_collate_fn


def train_loop(model, train_cfg, dataset_cfg, device):
    """
    YOLO Phase-1 training loop with:
    - AMP (fp16)
    - Progress bar
    - Best + Last checkpoint saving
    - Kaggle AUTO dataset support
    """

    logging.info("Starting YOLO training loop")

    # --------------------------------------------------
    # Dataset root
    # --------------------------------------------------
    yolo_root = dataset_cfg["yolo"]["root"]

    # AUTO support for Kaggle
    if yolo_root == "AUTO":
        yolo_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"

    logging.info(f"Using dataset root: {yolo_root}")

    # --------------------------------------------------
    # Experiment directory (model saving)
    # --------------------------------------------------
    exp_dir = Path("experiments/yolo_baseline")
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = exp_dir / "best.pt"
    last_ckpt = exp_dir / "last.pt"

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    train_dataset = YOLODataset(yolo_root, split="train")
    val_dataset = YOLODataset(yolo_root, split="val")

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    # --------------------------------------------------
    # DataLoader (IMPORTANT: custom collate_fn)
    # --------------------------------------------------
    dl_cfg = train_cfg["dataloader"]

    train_loader = DataLoader(
        train_dataset,
        batch_size=dl_cfg["batch_size"],
        shuffle=True,
        num_workers=dl_cfg["num_workers"],
        pin_memory=dl_cfg["pin_memory"],
        collate_fn=yolo_collate_fn,  # 🔥 REQUIRED for YOLO targets
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------
    optimizer = SGD(
        model.parameters(),
        lr=train_cfg["training"]["learning_rate"],
        momentum=train_cfg["training"]["momentum"],
        weight_decay=train_cfg["training"]["weight_decay"],
    )

    epochs = train_cfg["training"]["epochs"]
    scaler = GradScaler()

    best_loss = float("inf")

    # --------------------------------------------------
    # Training loop
    # --------------------------------------------------
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(
            train_loader,
            desc=f"Epoch [{epoch + 1}/{epochs}]",
            ncols=100,
        )

        for images, targets in pbar:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with autocast():
                loss_dict = model(images, targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch + 1}/{epochs} | Avg Loss: {avg_loss:.4f}")

        # --------------------------------------------------
        # Save LAST checkpoint
        # --------------------------------------------------
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
            },
            last_ckpt,
        )

        # --------------------------------------------------
        # Save BEST checkpoint
        # --------------------------------------------------
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": avg_loss,
                },
                best_ckpt,
            )
            logging.info(f"✅ Best model saved (loss={best_loss:.4f})")

    logging.info("✅ YOLO Phase-1 training completed")
