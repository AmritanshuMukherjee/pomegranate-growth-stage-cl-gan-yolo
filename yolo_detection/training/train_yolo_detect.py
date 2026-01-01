import os
import logging
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm import tqdm

from yolo_detection.models.yolo_detector import YOLODetector
from utils.data_loader import YOLODataset
from utils.config_loader import load_yaml


def resolve_dataset_root(cfg):
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    if cfg["dataset"]["yolo"]["root"] == "AUTO":
        return "data/yolo"
    return cfg["dataset"]["yolo"]["root"]


def main():
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_yaml("yolo_detection/config/yolo_detect.yaml")
    yolo_root = resolve_dataset_root(cfg)

    train_ds = YOLODataset(yolo_root, split="train")

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
        model_size="medium",
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=cfg["training"]["lr"])
    save_dir = cfg["training"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

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
                {"model_state_dict": model.state_dict()},
                f"{save_dir}/best.pt",
            )
            logging.info("✅ Best YOLO detector saved")


if __name__ == "__main__":
    main()
