import logging
import torch
from pathlib import Path

from models.yolo.yolo_base import YOLOBase
from utils.train_utils import train_loop
from utils.config_loader import load_yaml


def main():
    logging.basicConfig(level=logging.INFO)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")

    # Load configs
    train_cfg = load_yaml("config/yolo_train.yaml")
    dataset_cfg = train_cfg["dataset"]

    # Create model (NO img_size here)
    model = YOLOBase(
        num_classes=dataset_cfg["num_classes"]
    ).to(device)

    # Start training
    train_loop(
        model=model,
        train_cfg=train_cfg,
        dataset_cfg=dataset_cfg,
        device=device,
    )


if __name__ == "__main__":
    main()
