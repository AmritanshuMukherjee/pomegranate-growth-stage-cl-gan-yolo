import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import logging
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from models.hybrid.yolo_hybrid_detector import YOLOHybridDetector
from utils.data_loader import YOLODataset, yolo_collate_fn
from utils.config_loader import load_yaml


def main():
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    cfg = load_yaml("config/yolo_hybrid_train.yaml")

    yolo_root = cfg["dataset"]["yolo"]["root"]
    if yolo_root == "AUTO":
        yolo_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"

    logging.info(f"Using dataset root: {yolo_root}")

    train_ds = YOLODataset(yolo_root, split="train")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
        collate_fn=yolo_collate_fn,
    )

    model = YOLOHybridDetector(
        num_classes=cfg["dataset"]["num_classes"],
        classifier_ckpt="models/weights/yolo_classifier_best.pt",
        model_size="medium",
    ).to(device)

    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg["training"]["lr"],
        weight_decay=1e-4,
    )

    exp_dir = Path("experiments/yolo_hybrid")
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_loss = float("inf")
    best_ckpt = exp_dir / "best.pt"

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}]")
        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} | Loss={avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "loss": best_loss,
                },
                best_ckpt,
            )
            logging.info("✅ Best hybrid detector saved")

    logging.info("✅ Hybrid YOLO training completed")


if __name__ == "__main__":
    main()
