import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from yolo_detection.models.yolo_detector import YOLODetector
from yolo_detection.config.yolo_detect import load_config
from utils.data_loader import YOLODataset


def main():
    logging.basicConfig(level=logging.INFO)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    cfg = load_config("yolo_detection/config/yolo_detect.yaml")

    dataset_root = cfg["dataset"]["root"]
    if dataset_root == "AUTO":
        dataset_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"

    train_ds = YOLODataset(dataset_root, split="train")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_ds.collate_fn,
    )

    model = YOLODetector(
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"]
    )

    for epoch in range(cfg["training"]["epochs"]):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = loss_dict["total_loss"]

            loss.backward()
            optimizer.step()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

    logging.info("✅ Training completed")


if __name__ == "__main__":
    main()
