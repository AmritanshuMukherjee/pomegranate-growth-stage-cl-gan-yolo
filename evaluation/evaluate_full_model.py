# evaluation/evaluate_full_model.py
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pathlib import Path
from torch.utils.data import DataLoader

from models.yolo.yolo_attention import YOLOAttention
from utils.data_loader import YOLODataset
from utils.augmentations import get_val_transforms
from utils.metrics import calculate_map
from utils.config_loader import load_all_configs


def evaluate_full_model(checkpoint_path: str):
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

    preds, gts = [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            preds.extend(model.predict(imgs))
            gts.extend(targets.numpy())

    print("mAP@0.5:", calculate_map(preds, gts))


if __name__ == "__main__":
    import sys
    evaluate_full_model(sys.argv[1])
