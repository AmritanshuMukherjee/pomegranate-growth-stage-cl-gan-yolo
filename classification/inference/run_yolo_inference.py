"""
YOLO Baseline Inference Script
Uses model.predict() (correct for this codebase)
Draws bounding boxes, class labels, confidence scores
"""

import os
import sys
import cv2
import torch
import logging
import numpy as np
from pathlib import Path

# ---------------------------------------------------------------------
# Fix Python path
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.yolo.yolo_base import YOLOBase
from utils.config_loader import load_yaml

# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)

CLASS_NAMES = ["immature", "mature", "overripe"]

# ---------------------------------------------------------------------
def draw_detections(image, detections):
    h, w = image.shape[:2]

    for det in detections:
        x1, y1, x2, y2, conf, cls_id = det.tolist()
        cls_id = int(cls_id)

        label = f"{CLASS_NAMES[cls_id]} {conf:.2f}"

        x1, y1 = int(x1 * w), int(y1 * h)
        x2, y2 = int(x2 * w), int(y2 * h)

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            image,
            label,
            (x1, max(y1 - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
        )

    return image

# ---------------------------------------------------------------------
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    cfg = load_yaml(ROOT / "config" / "yolo_train.yaml")
    img_size = cfg["dataset"]["image_size"]
    num_classes = cfg["dataset"]["num_classes"]

    weights_path = ROOT / "models" / "weights" / "yolo_baseline_best.pt"
    if not weights_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    out_dir = ROOT / "experiments" / "yolo_baseline" / "inference"
    out_dir.mkdir(parents=True, exist_ok=True)

    model = YOLOBase(
        num_classes=num_classes,
        input_size=img_size,
        model_size="medium",
    )

    ckpt = torch.load(weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    img_dir = ROOT / "data" / "yolo" / "images" / "test"
    images = sorted(img_dir.glob("*.jpg"))[:20]

    logging.info(f"Running inference on {len(images)} images...")

    for img_path in images:
        img = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_size, img_size))

        tensor = (
            torch.from_numpy(img_resized)
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(device)

        with torch.no_grad():
            detections = model.predict(
                tensor,
                conf_threshold=0.25,
                iou_threshold=0.45,
            )

        if len(detections) == 0 or detections[0].numel() == 0:
            logging.warning(f"No detections for {img_path.name}")
            continue

        det = detections[0]

        # visualize top 5
        det = det[det[:, 4].argsort(descending=True)[:5]]

        vis_img = draw_detections(img.copy(), det)
        cv2.imwrite(str(out_dir / img_path.name), vis_img)

    logging.info(f"✅ Inference results saved to: {out_dir}")

# ---------------------------------------------------------------------
if __name__ == "__main__":
    main()
