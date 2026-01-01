"""
Merge real and synthetic datasets into YOLO-ready structure
Synthetic data is used ONLY for training
"""

import shutil
import random
from pathlib import Path
from tqdm import tqdm

# -----------------------------
# CONFIG
# -----------------------------
REAL_IMG_DIR = Path("data/raw/source_1/images")
REAL_LBL_DIR = Path("data/raw/source_1/labels")

SYN_IMG_DIR = Path("data/synthetic/gan_images")
SYN_LBL_DIR = Path("data/synthetic/gan_labels")

OUT_BASE = Path("data/yolo")

SPLITS = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

SEED = 42
random.seed(SEED)


def prepare_dirs():
    for split in SPLITS:
        (OUT_BASE / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_BASE / "labels" / split).mkdir(parents=True, exist_ok=True)


def split_real_images(images):
    random.shuffle(images)
    n = len(images)
    train_end = int(n * SPLITS["train"])
    val_end = train_end + int(n * SPLITS["val"])

    return {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }


def copy_pair(img_path, lbl_path, split):
    shutil.copy(img_path, OUT_BASE / "images" / split / img_path.name)
    shutil.copy(lbl_path, OUT_BASE / "labels" / split / lbl_path.name)


def main():
    prepare_dirs()

    real_images = list(REAL_IMG_DIR.glob("*.jpg"))
    splits = split_real_images(real_images)

    print("INFO | Copying REAL images")
    for split, imgs in splits.items():
        for img in tqdm(imgs, desc=f"Real → {split}"):
            lbl = REAL_LBL_DIR / f"{img.stem}.txt"
            if lbl.exists():
                copy_pair(img, lbl, split)

    print("INFO | Copying SYNTHETIC images (train only)")
    syn_images = list(SYN_IMG_DIR.glob("*.jpg"))
    for img in tqdm(syn_images, desc="Synthetic → train"):
        lbl = SYN_LBL_DIR / f"{img.stem}.txt"
        if lbl.exists():
            copy_pair(img, lbl, "train")

    print("INFO | Dataset merge completed")
    print(f"INFO | YOLO dataset ready at {OUT_BASE}")


if __name__ == "__main__":
    main()
