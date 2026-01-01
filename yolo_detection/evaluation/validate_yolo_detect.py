import os
import torch
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


def validate():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_yaml("yolo_detection/config/yolo_detect.yaml")
    yolo_root = resolve_dataset_root(cfg)

    dataset = YOLODataset(yolo_root, split="val")

    model = YOLODetector(
        num_classes=cfg["dataset"]["num_classes"],
        model_size="medium",
    ).to(device)

    ckpt_path = "yolo_detection/experiments/yolo_detect/best.pt"
    assert os.path.exists(ckpt_path), "Checkpoint not found"

    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    total = 0
    detected = 0

    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Validating"):
            image, targets = dataset[i]
            image = image.unsqueeze(0).to(device)

            preds = model(image)
            if preds and len(preds[0]) > 0:
                detected += 1
            total += 1

    print("\n=== YOLO Detection Validation ===")
    print(f"Images with detections: {detected}/{total}")


if __name__ == "__main__":
    validate()
