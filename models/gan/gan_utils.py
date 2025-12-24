"""
GAN Utility Functions
"""

import torch
import numpy as np
from typing import List, Tuple, Optional
from PIL import Image
import os


def generate_synthetic_images(
    generator: torch.nn.Module,
    source_images: List[str],
    output_dir: str,
    device: str = "cuda",
    num_samples: int = 1000,
    image_size: int = 256,
):
    """
    Generate synthetic images using trained generator
    
    Args:
        generator: Trained generator model
        source_images: List of source image paths
        output_dir: Output directory for generated images
        device: Device to use
        num_samples: Number of samples to generate
        image_size: Image size for generation
    """
    os.makedirs(output_dir, exist_ok=True)
    
    generator.eval()
    generator.to(device)
    
    # Load and preprocess images
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    with torch.no_grad():
        for i in range(num_samples):
            # Randomly select source image
            source_path = np.random.choice(source_images)
            source_img = Image.open(source_path).convert('RGB')
            source_tensor = transform(source_img).unsqueeze(0).to(device)
            
            # Generate
            generated = generator(source_tensor)
            
            # Denormalize and save
            generated = (generated + 1) / 2  # [-1, 1] -> [0, 1]
            generated = generated.squeeze(0).cpu()
            generated_img = transforms.ToPILImage()(generated)
            
            output_path = os.path.join(output_dir, f"generated_{i:06d}.jpg")
            generated_img.save(output_path)
            
            if (i + 1) % 100 == 0:
                print(f"Generated {i + 1}/{num_samples} images")


def create_paired_dataset(
    source_dir: str,
    target_dir: str,
    output_dir: str,
):
    """
    Create paired dataset for GAN training
    
    Args:
        source_dir: Directory with source images
        target_dir: Directory with target images
        output_dir: Output directory for paired dataset
    """
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    source_images = sorted([f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png'))])
    target_images = sorted([f for f in os.listdir(target_dir) if f.endswith(('.jpg', '.png'))])
    
    # Match images by name
    for source_img, target_img in zip(source_images, target_images):
        if source_img == target_img:
            source_path = os.path.join(source_dir, source_img)
            target_path = os.path.join(target_dir, target_img)
            
            output_source = os.path.join(output_dir, "source", source_img)
            output_target = os.path.join(output_dir, "target", target_img)
            
            os.makedirs(os.path.dirname(output_source), exist_ok=True)
            os.makedirs(os.path.dirname(output_target), exist_ok=True)
            
            shutil.copy(source_path, output_source)
            shutil.copy(target_path, output_target)

