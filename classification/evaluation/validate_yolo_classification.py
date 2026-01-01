import os
import torch
from tqdm import tqdm

from models.yolo.yolo_base import YOLOBase
from utils.data_loader import YOLODataset
from utils.config_loader import load_yaml


def resolve_dataset_root(cfg):
    if os.path.exists("/kaggle/input"):
        return "/kaggle/input/pomegranate-dataset/pomegranate_dataset/yolo"
    return "data/yolo"


def validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_yaml("config/yolo_train.yaml")
    num_classes = cfg["dataset"]["num_classes"]

    yolo_root = resolve_dataset_root(cfg)
    dataset = YOLODataset(yolo_root, split="val")

    model = YOLOBase(
        num_classes=num_classes,
        input_size=cfg["dataset"]["image_size"],
        model_size="medium",
    ).to(device)

    ckpt = torch.load("models/weights/yolo_baseline_best.pt", map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img, targets in tqdm(dataset, desc="Validating"):
            img = img.unsqueeze(0).to(device)

            preds = model.predict(img, conf_threshold=0.1)

            if preds is None or len(preds[0]) == 0:
                continue

            # 🔥 highest confidence detection
            best_pred = max(preds[0], key=lambda x: x[4])
            pred_cls = int(best_pred[5])

            # Ground truth class (only one object per image)
            gt_cls = int(targets[0][1].item())

            if pred_cls == gt_cls:
                correct += 1
            total += 1

    acc = correct / (total + 1e-6)

    print("\n=== Classification Validation ===")
    print(f"Accuracy: {acc:.4f} ({correct}/{total})")


if __name__ == "__main__":
    validate()