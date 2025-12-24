# training/train_full_pipeline.py
"""
Train full pipeline: GAN + Curriculum + Attention YOLO
"""
import sys
from pathlib import Path

# Add project root to PYTHONPATH
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from models.yolo.yolo_attention import YOLOAttention
from utils.data_loader import YOLODataset
from utils.curriculum_utils import CurriculumSampler, calculate_difficulty_scores
from utils.augmentations import get_train_transforms, get_val_transforms
from utils.metrics import calculate_map
from utils.logger import setup_logger
from utils.config_loader import load_all_configs


def train_full_pipeline():
    configs = load_all_configs()

    dataset_config = configs["dataset"]
    yolo_config = configs["yolo"]
    curriculum_config = configs["curriculum"]
    attention_config = configs["attention"]
    train_config = configs["training"]

    device = train_config.get("device", "cuda")
    exp_dir = Path(train_config["experiment_dir"]) / train_config["experiment_name"]
    ckpt_dir = exp_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(exp_dir / "logs/train.log")
    logger.info("🚀 Starting FULL PIPELINE training")

    model = YOLOAttention(
        num_classes=dataset_config["num_classes"],
        input_size=dataset_config["image_size"],
        model_size=yolo_config["model"]["size"],
        attention_type=attention_config["type"],
        attention_config=attention_config.get("cbam", {})
    ).to(device)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=yolo_config["model"]["training"]["learning_rate"],
        momentum=yolo_config["model"]["training"]["momentum"],
        weight_decay=yolo_config["model"]["training"]["weight_decay"],
    )

    train_ds = YOLODataset(
        Path(dataset_config["sources"]["source_1"]["path"]),
        "images",
        "labels",
        dataset_config["image_size"],
        get_train_transforms(dataset_config),
        "train",
    )

    val_ds = YOLODataset(
        Path(dataset_config["sources"]["source_1"]["path"]),
        "images",
        "labels",
        dataset_config["image_size"],
        get_val_transforms(),
        "val",
    )

    difficulty_scores = calculate_difficulty_scores(
        train_ds,
        curriculum_config["difficulty_metrics"],
        curriculum_config["selection"]["weights"]
    )

    best_map = 0.0
    global_epoch = 0

    for stage_name, stage in curriculum_config["stages"].items():
        logger.info(
            f"📚 Curriculum Stage: {stage_name} | "
            f"Difficulty: {stage['difficulty']} | "
            f"Epochs: {stage['epochs']}"
        )

        sampler = CurriculumSampler(
            train_ds, difficulty_scores, stage,
            curriculum_config["selection"]["method"]
        )

        train_loader = DataLoader(
            train_ds,
            batch_size=yolo_config["model"]["training"]["batch_size"],
            sampler=sampler,
            collate_fn=train_ds.collate_fn
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=yolo_config["model"]["training"]["batch_size"],
            shuffle=False,
            collate_fn=val_ds.collate_fn
        )

        for epoch in range(stage["epochs"]):
            global_epoch += 1
            model.train()

            pbar = tqdm(train_loader, desc=f"{stage_name} | Epoch {epoch+1}")
            for batch_idx, (imgs, targets) in enumerate(pbar):
                imgs, targets = imgs.to(device), targets.to(device)

                optimizer.zero_grad()
                loss_dict = model(imgs, targets)
                loss = loss_dict["total_loss"]
                loss.backward()
                optimizer.step()

                if batch_idx % 10 == 0:
                    pbar.set_postfix(loss=loss.item())
                    logger.info(
                        f"Epoch {global_epoch} | "
                        f"Batch {batch_idx}/{len(train_loader)} | "
                        f"Loss: {loss.item():.4f}"
                    )

            model.eval()
            preds, gts = [], []
            with torch.no_grad():
                for imgs, targets in val_loader:
                    imgs = imgs.to(device)
                    preds.extend(model.predict(imgs))
                    gts.extend(targets.numpy())

            map50 = calculate_map(preds, gts)
            logger.info(f"📊 Epoch {global_epoch} mAP@0.5 = {map50:.4f}")

            if map50 > best_map:
                best_map = map50
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")
                logger.info("✅ Saved new BEST model")

    logger.info("🎉 Full pipeline training completed")


if __name__ == "__main__":
    train_full_pipeline()
