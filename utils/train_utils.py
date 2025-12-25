import logging
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from utils.data_loader import YOLODataset
from utils.augmentations import get_train_transforms, get_val_transforms


def resolve_yolo_root():
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        dataset_dir = list(kaggle_input.iterdir())[0]
        yolo_root = dataset_dir / "data" / "yolo"
        if not yolo_root.exists():
            raise RuntimeError("data/yolo not found inside Kaggle dataset")
        return yolo_root
    return Path("data/yolo")


def train_loop(model, train_cfg, dataset_cfg, device):
    logging.info("Starting YOLO training loop")

    yolo_root = resolve_yolo_root()

    train_dataset = YOLODataset(
        yolo_root=yolo_root,
        split="train",
        transform=get_train_transforms(dataset_cfg),
    )

    val_dataset = YOLODataset(
        yolo_root=yolo_root,
        split="val",
        transform=get_val_transforms(dataset_cfg),
    )

    logging.info(f"Train samples: {len(train_dataset)}")
    logging.info(f"Val samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["batch_size"],
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        collate_fn=train_dataset.collate_fn,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )

    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    for epoch in range(train_cfg["epochs"]):
        model.train()
        epoch_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{train_cfg['epochs']}]")

        for images, targets in pbar:
            images = images.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
                loss_dict = model(images, targets)
                loss = loss_dict["total_loss"]

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        avg_loss = epoch_loss / len(train_loader)
        logging.info(f"Epoch [{epoch+1}/{train_cfg['epochs']}] | Loss: {avg_loss:.4f}")

    logging.info("✅ Training completed")
