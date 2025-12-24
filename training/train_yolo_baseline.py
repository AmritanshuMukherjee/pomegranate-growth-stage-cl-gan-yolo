import sys
from pathlib import Path

# 🔥 Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import logging
import torch

from models.yolo.yolo_base import YOLOBase
from utils.config_loader import load_yaml
from utils.train_utils import train_loop


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    logging.info("🚀 Phase-1 | Training YOLO Baseline")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    train_cfg = load_yaml("config/yolo_train.yaml")

    # ✅ FIX: pass ONLY dataset section
    dataset_cfg = train_cfg["dataset"]

    model = YOLOBase(
        num_classes=dataset_cfg["num_classes"]
    ).to(device)

    train_loop(
        model=model,
        train_cfg=train_cfg,
        dataset_cfg=dataset_cfg,   # ✅ CORRECT
        device=device,
    )


if __name__ == "__main__":
    main()
