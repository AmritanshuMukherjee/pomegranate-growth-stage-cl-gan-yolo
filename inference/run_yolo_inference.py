import os
import cv2
import torch
import yaml
from pathlib import Path

from models.yolo.yolo_base import YOLOBase


# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
WEIGHTS_PATH = "models/weights/yolo_baseline_best.pt"
DATASET_ROOT = "data/yolo"  # change if needed
SPLIT = "val"               # val or test
CONF_THRESH = 0.25
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAVE_DIR = Path("experiments/yolo_baseline/inference")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------
def load_model():
    model = YOLOBase(num_classes=3)
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


# ---------------------------------------------------------
# Inference
# ---------------------------------------------------------
def run_inference():
    model = load_model()

    images_dir = os.path.join(DATASET_ROOT, "images", SPLIT)
    image_files = sorted(os.listdir(images_dir))[:20]  # first 20 samples

    print(f"Running inference on {len(image_files)} images...")

    with torch.no_grad():
        for img_name in image_files:
            img_path = os.path.join(images_dir, img_name)
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
            img_tensor = img_tensor.unsqueeze(0).to(DEVICE)

            outputs = model(img_tensor, targets=None)

            # TODO: decode predictions (simplified visualization placeholder)
            save_path = SAVE_DIR / img_name
            cv2.imwrite(str(save_path), img)

    print(f"✅ Inference results saved to: {SAVE_DIR}")


if __name__ == "__main__":
    run_inference()
