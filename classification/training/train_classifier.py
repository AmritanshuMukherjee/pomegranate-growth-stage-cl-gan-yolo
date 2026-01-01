import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import os
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.hybrid.yolo_classifier import YOLOClassifier
from utils.data_loader import YOLOClassificationDataset
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
    logging.info(f"Using device: {device}")

    cfg = load_yaml("config/yolo_train.yaml")
    yolo_root = resolve_dataset_root(cfg)

    # --------------------------------------------------
    # Dataset
    # --------------------------------------------------
    dataset = YOLOClassificationDataset(
        yolo_root,
        split="train"
    )

    loader = DataLoader(
        dataset,
        batch_size=16,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------
    model = YOLOClassifier(
        num_classes=cfg["dataset"]["num_classes"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # --------------------------------------------------
    # Save paths
    # --------------------------------------------------
    exp_dir = Path("experiments/yolo_classifier")
    exp_dir.mkdir(parents=True, exist_ok=True)

    best_ckpt = exp_dir / "best.pt"
    best_acc = 0.0

    # --------------------------------------------------
    # Training
    # --------------------------------------------------
    for epoch in range(30):
        model.train()
        correct = total = 0
        epoch_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}")

        for images, labels in pbar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            epoch_loss += loss.item()

            pbar.set_postfix(loss=f"{loss.item():.4f}")

        acc = correct / total
        logging.info(f"Epoch {epoch+1} | Acc={acc:.4f}")

        # SAVE BEST
        if acc > best_acc:
            best_acc = acc
            torch.save(
                {
                    "epoch": epoch + 1,
                    "accuracy": best_acc,
                    "model_state_dict": model.state_dict(),
                },
                best_ckpt,
            )
            logging.info(f"✅ Best model saved (acc={best_acc:.4f})")

    logging.info("✅ Classification training complete")


if __name__ == "__main__":
    main()
