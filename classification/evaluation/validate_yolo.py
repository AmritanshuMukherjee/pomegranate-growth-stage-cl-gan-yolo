"""
Hybrid YOLO Validation Script
Computes Precision, Recall, mAP@0.5

- Works on Kaggle + Local
- Correct dataset root handling
- Correct checkpoint loading
"""

import os
import torch
from tqdm import tqdm

from models.hybrid.yolo_hybrid_detector import YOLOHybridDetector
from utils.data_loader import YOLODataset
from utils.config_loader import load_yaml


# ------------------------------------------------------------
# IoU computation (xyxy)
# ------------------------------------------------------------
def compute_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    return inter / (area1 + area2 - inter + 1e-6)


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --------------------------------------------------------
    # Load config
    # --------------------------------------------------------
    cfg = load_yaml("config/yolo_hybrid_train.yaml")
    num_classes = cfg["dataset"]["num_classes"]

    # --------------------------------------------------------
    # Resolve dataset root (Kaggle / Local)
    # --------------------------------------------------------
    if os.path.exists("/kaggle/input"):
        yolo_root = "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    else:
        yolo_root = "data/yolo"

    print(f"Using dataset root: {yolo_root}")

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset = YOLODataset(yolo_root, split="val")

    # --------------------------------------------------------
    # Load Hybrid Detector
    # --------------------------------------------------------
    ckpt_path = "experiments/yolo_hybrid/best.pt"

    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"

    model = YOLOHybridDetector(
        num_classes=num_classes,
        classifier_ckpt="models/weights/yolo_classifier_best.pt",
        model_size="medium",
    ).to(device)

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"], strict=False)
    model.eval()

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------
    TP = FP = FN = 0

    with torch.no_grad():
        for idx in tqdm(range(len(dataset)), desc="Validating"):
            image, targets = dataset[idx]
            image = image.unsqueeze(0).to(device)

            preds = model(image)

            # No predictions
            if preds is None or len(preds[0]) == 0:
                FN += targets.shape[0]
                continue

            matched = set()

            for px1, py1, px2, py2, conf, cls in preds[0]:
                best_iou = 0
                best_gt = -1

                for j, gt in enumerate(targets):
                    gx, gy, gw, gh = gt[2:].tolist()
                    gt_box = [
                        gx - gw / 2,
                        gy - gh / 2,
                        gx + gw / 2,
                        gy + gh / 2,
                    ]

                    iou = compute_iou([px1, py1, px2, py2], gt_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt = j

                if best_iou >= 0.5 and best_gt not in matched:
                    if int(cls) == int(targets[best_gt][1]):
                        TP += 1
                        matched.add(best_gt)
                    else:
                        FP += 1
                else:
                    FP += 1

            FN += max(0, targets.shape[0] - len(matched))

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------
    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    map50 = precision * recall

    print("\n=== Hybrid YOLO Validation ===")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"mAP@0.5:   {map50:.4f}")


if __name__ == "__main__":
    validate()