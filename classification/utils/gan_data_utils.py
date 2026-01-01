"""
GAN Data Utilities
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np
from PIL import Image


def prepare_gan_dataset(
    source_dir: str,
    target_dir: str,
    output_dir: str,
    image_size: int = 256,
):
    """
    Prepare paired dataset for GAN training
    
    Args:
        source_dir: Directory with source images
        target_dir: Directory with target images
        output_dir: Output directory for paired dataset
        image_size: Resize images to this size
    """
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'source').mkdir(exist_ok=True)
    (output_dir / 'target').mkdir(exist_ok=True)
    
    # Get image files
    source_images = sorted(
        list(source_dir.glob('*.jpg')) +
        list(source_dir.glob('*.png')) +
        list(source_dir.glob('*.JPG')) +
        list(source_dir.glob('*.PNG'))
    )
    
    target_images = sorted(
        list(target_dir.glob('*.jpg')) +
        list(target_dir.glob('*.png')) +
        list(target_dir.glob('*.JPG')) +
        list(target_dir.glob('*.PNG'))
    )
    
    # Match images by filename
    matched_pairs = []
    for source_img in source_images:
        target_img = target_dir / source_img.name
        if target_img.exists():
            matched_pairs.append((source_img, target_img))
    
    print(f"Found {len(matched_pairs)} matched image pairs")
    
    # Copy and resize images
    for idx, (source_img, target_img) in enumerate(matched_pairs):
        # Load and resize
        source = Image.open(source_img).convert('RGB')
        target = Image.open(target_img).convert('RGB')
        
        source = source.resize((image_size, image_size), Image.LANCZOS)
        target = target.resize((image_size, image_size), Image.LANCZOS)
        
        # Save
        output_source = output_dir / 'source' / f"{idx:06d}.jpg"
        output_target = output_dir / 'target' / f"{idx:06d}.jpg"
        
        source.save(output_source, 'JPEG', quality=95)
        target.save(output_target, 'JPEG', quality=95)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(matched_pairs)} pairs")
    
    print(f"Dataset prepared: {output_dir}")


def create_synthetic_labels(
    synthetic_images_dir: str,
    output_labels_dir: str,
    source_labels_dir: str,
):
    """
    Create labels for synthetic images (copy from source)
    
    Args:
        synthetic_images_dir: Directory with synthetic images
        output_labels_dir: Output directory for labels
        source_labels_dir: Source directory with original labels
    """
    synthetic_images_dir = Path(synthetic_images_dir)
    output_labels_dir = Path(output_labels_dir)
    source_labels_dir = Path(source_labels_dir)
    
    output_labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Get synthetic images
    synthetic_images = list(synthetic_images_dir.glob('*.jpg')) + list(synthetic_images_dir.glob('*.png'))
    
    # Copy labels from source (matching by index or name)
    for idx, img_path in enumerate(synthetic_images):
        # Try to find matching label
        label_name = img_path.stem + '.txt'
        source_label = source_labels_dir / label_name
        
        if source_label.exists():
            shutil.copy(source_label, output_labels_dir / label_name)
        else:
            # Create empty label if not found
            with open(output_labels_dir / label_name, 'w') as f:
                pass  # Empty file
    
    print(f"Created labels for {len(synthetic_images)} synthetic images")

