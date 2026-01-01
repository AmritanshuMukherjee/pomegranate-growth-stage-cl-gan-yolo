import os
import sys
import logging
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------------------------
# FIX PYTHON PATH (ABSOLUTE, SAFE)
# -------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# -------------------------------------------------
# IMPORTS (NOW WORK CORRECTLY)
# -------------------------------------------------
from yolo_detection.models.yolo_detector import YOLODetector
from utils.data_loader import YOLODataset
from utils.config_loader import load_yaml

# -------------------------------------------------
# TRAINING
# -------------------------------------------------
def main():
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    cfg = load_yaml("yolo_detection/config/yolo_detect.yaml")

    dataset_root = cfg["dataset"]["root"]
    if dataset_root == "AUTO":
        dataset_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"

    logging.info(f"Using dataset root: {dataset_root}")

    train_ds = YOLODataset(dataset_root, split="train")
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["dataloader"]["batch_size"],
        shuffle=True,
        num_workers=cfg["dataloader"]["num_workers"],
        pin_memory=True,
        collate_fn=train_ds.collate_fn,
    )

    model = YOLODetector(
        num_classes=cfg["dataset"]["num_classes"],
        model_size=cfg["model"]["size"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=1e-4,
    )

    os.makedirs("experiments/yolo_detect", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg['training']['epochs']}]")
        for images, targets in pbar:
            images, targets = images.to(device), targets.to(device)

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
                    "loss": avg_loss,
                },
                "experiments/yolo_detect/best.pt",
            )
            logging.info("✅ Best detector saved")

    logging.info("✅ YOLO Detection training completed")


if __name__ == "__main__":
    main()
