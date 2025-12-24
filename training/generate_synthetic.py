"""
Source-conditioned GAN synthetic image generation with YOLO label preservation
"""

import sys
from pathlib import Path

# -------------------------------------------------
# FIX PYTHON PATH (CRITICAL)
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# -------------------------------------------------
import yaml
import torch
import argparse
import cv2
from tqdm import tqdm

from models.gan.generator import UNetGenerator


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def generate_synthetic(gan_cfg_path, dataset_cfg_path, checkpoint_path):
    print("INFO | Loading configs")

    gan_cfg = load_yaml(gan_cfg_path)["gan"]
    dataset_cfg = load_yaml(dataset_cfg_path)["dataset"]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # -----------------------
    # Load generator
    # -----------------------
    gcfg = gan_cfg["generator"]
    generator = UNetGenerator(
        input_channels=gcfg["input_channels"],
        output_channels=gcfg["output_channels"],
        base_filters=gcfg["base_filters"],
        num_downsampling=gcfg["num_downsampling"],
        num_upsampling=gcfg["num_upsampling"],
        use_dropout=gcfg["use_dropout"],
        dropout_rate=gcfg["dropout_rate"],
        use_batch_norm=gcfg["use_batch_norm"],
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # -----------------------
    # Dataset paths
    # -----------------------
    src = dataset_cfg["sources"]["source_1"]
    base_path = Path(src["path"])
    img_dir = base_path / src["images"]
    lbl_dir = base_path / src["labels"]

    out_img_dir = Path(gan_cfg["generation"]["output_dir"])
    out_lbl_dir = Path(gan_cfg["generation"]["labels_dir"])
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    num_variants = gan_cfg["generation"]["num_samples_per_class"]

    image_files = list(img_dir.glob("*.jpg"))
    print(f"INFO | Found {len(image_files)} real images")

    # -----------------------
    # Generate synthetic data
    # -----------------------
    with torch.no_grad():
        for img_path in tqdm(image_files, desc="Generating synthetic images"):
            label_path = lbl_dir / f"{img_path.stem}.txt"
            if not label_path.exists():
                continue

            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (256, 256))
            img = img.astype("float32") / 255.0
            img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

            for i in range(num_variants):
                fake = generator(img).squeeze(0)
                fake = (fake + 1) / 2
                fake = fake.clamp(0, 1)

                out_name = f"{img_path.stem}_syn_{i}"

                fake_img = (
                    fake.permute(1, 2, 0).cpu().numpy() * 255
                ).astype("uint8")

                cv2.imwrite(
                    str(out_img_dir / f"{out_name}.jpg"),
                    cv2.cvtColor(fake_img, cv2.COLOR_RGB2BGR),
                )

                with open(label_path, "r") as src_lbl, open(
                    out_lbl_dir / f"{out_name}.txt", "w"
                ) as dst_lbl:
                    dst_lbl.write(src_lbl.read())

    print("INFO | Synthetic image generation completed")
    print(f"INFO | Images → {out_img_dir}")
    print(f"INFO | Labels → {out_lbl_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gan-config", required=True)
    parser.add_argument("--dataset-config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    generate_synthetic(
        args.gan_config,
        args.dataset_config,
        args.checkpoint,
    )
