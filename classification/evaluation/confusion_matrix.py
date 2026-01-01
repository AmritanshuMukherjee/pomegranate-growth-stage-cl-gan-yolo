# evaluation/confusion_matrix.py
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix

from models.yolo.yolo_attention import YOLOAttention
from utils.data_loader import YOLODataset
from utils.augmentations import get_val_transforms
from utils.metrics import calculate_iou, convert_to_xyxy
from utils.config_loader import load_all_configs


def generate_confusion_matrix(checkpoint_path: str, output_path: str):
    configs = load_all_configs()

    dataset_config = configs["dataset"]
    yolo_config = configs["yolo"]
    attention_config = configs["attention"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = YOLOAttention(
        num_classes=dataset_config["num_classes"],
        input_size=dataset_config["image_size"],
        model_size=yolo_config["model"]["size"],
        attention_type=attention_config["type"],
        attention_config=attention_config["cbam"],
        attention_placement=attention_config["placement"],
    ).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    dataset = YOLODataset(
        Path(dataset_config["sources"]["source_1"]["path"]),
        "images",
        "labels",
        dataset_config["image_size"],
        get_val_transforms(),
        "test",
    )

    loader = DataLoader(dataset, batch_size=1, collate_fn=dataset.collate_fn)

    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds = model.predict(imgs)[0]

            targets = targets[0].numpy()
            preds = preds.cpu().numpy() if isinstance(preds, torch.Tensor) else preds

            for t in targets:
                if t[0] < 0:
                    continue

                true_cls = int(t[0])
                best_iou, best_cls = 0, true_cls

                for p in preds:
                    iou = calculate_iou(
                        convert_to_xyxy(p[:4]),
                        convert_to_xyxy(t[1:5]),
                    )
                    if iou > best_iou and iou > 0.5:
                        best_iou = iou
                        best_cls = int(p[5])

                y_true.append(true_cls)
                y_pred.append(best_cls)

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=dataset_config["class_names"],
        yticklabels=dataset_config["class_names"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)


if __name__ == "__main__":
    import sys
    generate_confusion_matrix(sys.argv[1], sys.argv[2])
