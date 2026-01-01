import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
import os
import torch
import logging
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from yolo_detection.models.yolo_detector import YOLODetector
from yolo_detection.utils.yolo_dataset import YOLODetectionDataset
import yaml


def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_dataset_root():
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    return "data/yolo"


def main():
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    cfg = load_cfg("yolo_detection/config/yolo_detect.yaml")
    dataset_root = resolve_dataset_root()
    logging.info(f"Using dataset root: {dataset_root}")

    train_ds = YOLODetectionDataset(
        dataset_root,
        split="train",
        img_size=cfg["dataset"]["image_size"]
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=lambda x: tuple(zip(*x))
    )

    model = YOLODetector(
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])

    os.makedirs("yolo_detection/experiments/yolo_detect", exist_ok=True)
    best_loss = float("inf")

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{cfg['training']['epochs']}]")

        for imgs, targets in pbar:
            imgs = torch.stack(imgs).to(device)
            targets = [t.to(device) for t in targets]

            optimizer.zero_grad()
            loss_dict = model(imgs, targets)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch {epoch+1} | Loss: {avg_loss:.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(
                {"model_state_dict": model.state_dict()},
                "yolo_detection/experiments/yolo_detect/best.pt"
            )
            logging.info("✅ Best YOLO detector saved")


if __name__ == "__main__":
    main()
