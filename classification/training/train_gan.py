"""
Train GAN for data augmentation (Pix2Pix-style)
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import yaml
import torch
import argparse
from pathlib import Path
from torch.utils.data import DataLoader

from models.gan.generator import UNetGenerator
from models.gan.discriminator import PatchGANDiscriminator
from models.gan.gan_trainer import GANTrainer
from utils.data_loader import GANDataset
from utils.logger import setup_logger


def load_yaml(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def train_gan(gan_config_path: str, dataset_config_path: str):
    # =====================
    # LOAD CONFIGS
    # =====================
    gan_cfg = load_yaml(gan_config_path)["gan"]
    dataset_cfg = load_yaml(dataset_config_path)["dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    experiment_dir = Path("experiments/gan_training")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(experiment_dir / "logs/train.log")
    logger.info("Starting GAN training")

    # =====================
    # MODELS
    # =====================
    generator = UNetGenerator(
        input_channels=gan_cfg["generator"]["input_channels"],
        output_channels=gan_cfg["generator"]["output_channels"],
        base_filters=gan_cfg["generator"]["base_filters"],
        num_downsampling=gan_cfg["generator"]["num_downsampling"],
        num_upsampling=gan_cfg["generator"]["num_upsampling"],
        use_dropout=gan_cfg["generator"]["use_dropout"],
        dropout_rate=gan_cfg["generator"]["dropout_rate"],
        use_batch_norm=gan_cfg["generator"]["use_batch_norm"],
    )

    discriminator = PatchGANDiscriminator(
        input_channels=gan_cfg["discriminator"]["input_channels"],
        base_filters=gan_cfg["discriminator"]["base_filters"],
        num_layers=gan_cfg["discriminator"]["num_layers"],
        use_batch_norm=gan_cfg["discriminator"]["use_batch_norm"],
    )

    trainer = GANTrainer(
        generator=generator,
        discriminator=discriminator,
        device=device,
        generator_lr=gan_cfg["training"]["generator_lr"],
        discriminator_lr=gan_cfg["training"]["discriminator_lr"],
        beta1=gan_cfg["training"]["beta1"],
        beta2=gan_cfg["training"]["beta2"],
        gan_loss_weight=gan_cfg["training"]["gan_loss_weight"],
        l1_loss_weight=gan_cfg["training"]["l1_loss_weight"],
    )

    # =====================
    # DATASET (FIXED)
    # =====================
    image_dir = Path(
        dataset_cfg["sources"]["source_1"]["path"]
    ) / dataset_cfg["sources"]["source_1"]["images"]

    dataset = GANDataset(
        source_dir=str(image_dir),
        target_dir=str(image_dir),
        image_size=gan_cfg["augmentation"]["resize"],
    )

    if len(dataset) == 0:
        raise RuntimeError(
            f"No images found for GAN training in: {image_dir}"
        )

    loader = DataLoader(
        dataset,
        batch_size=gan_cfg["training"]["batch_size"],
        shuffle=True,
        num_workers=0,  # Windows-safe
    )

    # =====================
    # TRAINING LOOP
    # =====================
    epochs = gan_cfg["training"]["epochs"]
    ckpt_dir = experiment_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    for epoch in range(epochs):
        losses = {"d": 0.0, "g": 0.0}

        for i, (real, target) in enumerate(loader):
            loss = trainer.train_step(real, target)

            losses["d"] += loss["loss_d"]
            losses["g"] += loss["loss_g"]

            if i % 10 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{epochs}] "
                    f"Batch [{i}/{len(loader)}] "
                    f"D={loss['loss_d']:.4f} G={loss['loss_g']:.4f}"
                )

        logger.info(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Avg D={losses['d']/len(loader):.4f} "
            f"Avg G={losses['g']/len(loader):.4f}"
        )

        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                ckpt_dir / f"checkpoint_epoch_{epoch+1}.pth"
            )
            logger.info("Checkpoint saved")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    args = parser.parse_args()

    train_gan(args.gan_config, args.dataset_config)
