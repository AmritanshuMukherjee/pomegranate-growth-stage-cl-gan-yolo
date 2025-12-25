import os
import sys

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


import logging
import torch

from models.yolo.yolo_base import YOLOBase
from utils.config_loader import load_yaml
from utils.train_utils import train_loop


def main():
    logging.basicConfig(level=logging.INFO)

    train_cfg = load_yaml("config/yolo_train.yaml")
    dataset_cfg = train_cfg["dataset"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

  model = YOLOBase(
    num_classes=dataset_cfg["num_classes"]
).to(device)

    train_loop(model, train_cfg["training"], dataset_cfg, device)


if __name__ == "__main__":
    main()
