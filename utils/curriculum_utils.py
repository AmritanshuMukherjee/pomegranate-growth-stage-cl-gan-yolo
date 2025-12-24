"""
Curriculum Learning Utilities
"""

import numpy as np
import torch
from torch.utils.data import Sampler
from typing import List, Dict, Optional
from pathlib import Path
import cv2


class CurriculumSampler(Sampler):
    """
    Sampler for curriculum learning
    """
    
    def __init__(
        self,
        dataset,
        difficulty_scores: np.ndarray,
        stage_config: Dict,
        method: str = 'scoring',
    ):
        self.dataset = dataset
        self.difficulty_scores = difficulty_scores
        self.stage_config = stage_config
        self.method = method
        
        # Get indices for current stage
        self.indices = self._get_stage_indices()
    
    def _get_stage_indices(self) -> List[int]:
        """Get indices for current curriculum stage"""
        difficulty = self.stage_config.get('difficulty', 'low')
        percentage = self.stage_config.get('percentage', 1.0)
        
        # Sort by difficulty
        sorted_indices = np.argsort(self.difficulty_scores)
        
        if difficulty == 'low':
            # Easy samples (low difficulty scores)
            n_samples = int(len(sorted_indices) * percentage)
            indices = sorted_indices[:n_samples].tolist()
        elif difficulty == 'medium':
            # Medium samples
            start = int(len(sorted_indices) * 0.3)
            end = int(len(sorted_indices) * (0.3 + percentage))
            indices = sorted_indices[start:end].tolist()
        else:  # high
            # Hard samples (high difficulty scores)
            n_samples = int(len(sorted_indices) * percentage)
            indices = sorted_indices[-n_samples:].tolist()
        
        return indices
    
    def __iter__(self):
        # Shuffle indices
        indices = self.indices.copy()
        np.random.shuffle(indices)
        return iter(indices)
    
    def __len__(self):
        return len(self.indices)


def calculate_difficulty_scores(
    dataset,
    metrics: List[str],
    weights: Dict[str, float],
) -> np.ndarray:
    """
    Calculate difficulty scores for each sample
    
    Args:
        dataset: Dataset object
        metrics: List of difficulty metrics to use
        weights: Weights for each metric
    
    Returns:
        Array of difficulty scores
    """
    scores = np.zeros(len(dataset))
    
    for idx in range(len(dataset)):
        image, targets = dataset[idx]
        
        sample_scores = {}
        
        # Object size
        if 'object_size' in metrics:
            if len(targets) > 0 and targets[0][0] >= 0:
                # Calculate average object size
                sizes = []
                for target in targets:
                    if target[0] >= 0:
                        w, h = target[3], target[4]
                        size = w * h
                        sizes.append(size)
                avg_size = np.mean(sizes) if sizes else 0.0
                sample_scores['object_size'] = 1.0 - avg_size  # Smaller = harder
            else:
                sample_scores['object_size'] = 0.5
        
        # Occlusion level (simplified - based on number of objects)
        if 'occlusion_level' in metrics:
            num_objects = sum(1 for t in targets if t[0] >= 0)
            sample_scores['occlusion_level'] = min(num_objects / 10.0, 1.0)  # More objects = harder
        
        # Lighting quality (simplified - based on image brightness variance)
        if 'lighting_quality' in metrics:
            if isinstance(image, torch.Tensor):
                img_np = image.permute(1, 2, 0).numpy()
            else:
                img_np = image
            brightness_var = np.var(img_np.mean(axis=2))  # Variance of mean brightness
            sample_scores['lighting_quality'] = min(brightness_var * 10, 1.0)  # Higher variance = harder
        
        # Background complexity (simplified - based on edge density)
        if 'background_complexity' in metrics:
            if isinstance(image, torch.Tensor):
                img_np = (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            else:
                img_np = (image * 255).astype(np.uint8)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
            sample_scores['background_complexity'] = edge_density
        
        # Label confidence (simplified - assume all labels are confident)
        if 'label_confidence' in metrics:
            sample_scores['label_confidence'] = 0.5  # Placeholder
        
        # Weighted sum
        total_score = 0.0
        total_weight = 0.0
        for metric, weight in weights.items():
            if metric in sample_scores:
                total_score += sample_scores[metric] * weight
                total_weight += weight
        
        scores[idx] = total_score / (total_weight + 1e-7)
    
    # Normalize scores to [0, 1]
    scores = (scores - scores.min()) / (scores.max() - scores.min() + 1e-7)
    
    return scores

