"""
Organize the raw dataset into the expected structure
"""

import shutil
from pathlib import Path

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Simple progress indicator without tqdm
    def tqdm(iterable, desc=None, **kwargs):
        if desc:
            print(desc)
        return iterable


def organize_dataset(
    raw_dataset_dir: str = "data/raw_dataset/Pomegranate Images Dataset/VOC2007",
    output_dir: str = "data/raw/source_1",
    use_splits: bool = True
):
    """
    Organize dataset from VOC2007 format to expected structure
    
    Args:
        raw_dataset_dir: Path to raw VOC2007 dataset
        output_dir: Output directory for organized dataset
        use_splits: Whether to use existing train/val/test splits
    """
    raw_path = Path(raw_dataset_dir)
    output_path = Path(output_dir)
    
    # Create output directories
    images_dir = output_path / "images"
    labels_dir = output_path / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths to source data
    source_images = raw_path / "JPEGImages"
    source_labels = raw_path / "labels"
    
    if use_splits:
        # Use existing splits
        splits_dir = raw_path / "ImageSets" / "Main"
        
        for split_name in ['train', 'val', 'test']:
            split_file = splits_dir / f"{split_name}.txt"
            
            if not split_file.exists():
                print(f"Warning: {split_file} not found, skipping {split_name} split")
                continue
            
            print(f"\nProcessing {split_name} split...")
            
            # Read image IDs
            with open(split_file, 'r') as f:
                image_ids = [line.strip() for line in f.readlines()]
            
            # Copy images and labels
            for img_id in tqdm(image_ids, desc=f"Copying {split_name}"):
                # Copy image
                src_image = source_images / f"{img_id}.jpg"
                if src_image.exists():
                    dst_image = images_dir / f"{img_id}.jpg"
                    shutil.copy2(src_image, dst_image)
                
                # Copy label
                src_label = source_labels / f"{img_id}.txt"
                if src_label.exists():
                    dst_label = labels_dir / f"{img_id}.txt"
                    shutil.copy2(src_label, dst_label)
            
            print(f"  Copied {len(image_ids)} {split_name} samples")
        
        # Create split files in data/splits
        splits_output = Path("data/splits")
        splits_output.mkdir(parents=True, exist_ok=True)
        
        for split_name in ['train', 'val', 'test']:
            split_file = splits_dir / f"{split_name}.txt"
            if split_file.exists():
                shutil.copy2(split_file, splits_output / f"{split_name}.txt")
    
    else:
        # Copy all images and labels
        print("Copying all images and labels (not using splits)...")
        
        # Get all image files
        image_files = list(source_images.glob("*.jpg"))
        
        for img_file in tqdm(image_files, desc="Copying files"):
            img_id = img_file.stem
            
            # Copy image
            dst_image = images_dir / img_file.name
            shutil.copy2(img_file, dst_image)
            
            # Copy label
            src_label = source_labels / f"{img_id}.txt"
            if src_label.exists():
                dst_label = labels_dir / f"{img_id}.txt"
                shutil.copy2(src_label, dst_label)
        
        print(f"  Copied {len(image_files)} images and labels")
    
    print(f"\nDataset organized successfully!")
    print(f"Images: {images_dir}")
    print(f"Labels: {labels_dir}")
    
    # Count files
    num_images = len(list(images_dir.glob("*.jpg")))
    num_labels = len(list(labels_dir.glob("*.txt")))
    print(f"\nTotal images: {num_images}")
    print(f"Total labels: {num_labels}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Organize dataset from VOC2007 format")
    parser.add_argument(
        "--raw-dir",
        type=str,
        default="data/raw_dataset/Pomegranate Images Dataset/VOC2007",
        help="Path to raw VOC2007 dataset"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw/source_1",
        help="Output directory for organized dataset"
    )
    parser.add_argument(
        "--no-splits",
        action="store_true",
        help="Don't use existing train/val/test splits (copy all files)"
    )
    
    args = parser.parse_args()
    
    organize_dataset(
        raw_dataset_dir=args.raw_dir,
        output_dir=args.output_dir,
        use_splits=not args.no_splits
    )
