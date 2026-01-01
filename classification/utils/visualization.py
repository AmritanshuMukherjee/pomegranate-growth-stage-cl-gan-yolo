"""
Visualization utilities
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import torch


def draw_boxes(
    image: np.ndarray,
    boxes: List[Tuple[float, float, float, float]],
    labels: List[int],
    scores: Optional[List[float]] = None,
    class_names: Optional[List[str]] = None,
    colors: Optional[dict] = None,
) -> np.ndarray:
    """
    Draw bounding boxes on image
    
    Args:
        image: Input image [H, W, 3]
        boxes: List of boxes [(x1, y1, x2, y2), ...]
        labels: List of class labels
        scores: Optional list of confidence scores
        class_names: Optional list of class names
        colors: Optional dict mapping class_id to color
    
    Returns:
        Image with drawn boxes
    """
    img = image.copy()
    
    # Default colors
    if colors is None:
        colors = {
            0: (255, 0, 0),      # bud - red
            1: (0, 255, 0),      # flower - green
            2: (255, 255, 0),    # fruit_immature - yellow
            3: (0, 0, 255),      # fruit_mature - blue
            4: (255, 0, 255),    # harvest - magenta
        }
    
    # Default class names
    if class_names is None:
        class_names = ['bud', 'flower', 'fruit_immature', 'fruit_mature', 'harvest']
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        x1, y1, x2, y2 = map(int, box)
        color = colors.get(int(label), (255, 255, 255))
        
        # Draw box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label_text = class_names[int(label)]
        if scores is not None:
            label_text += f" {scores[i]:.2f}"
        
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
        )
        
        # Draw label background
        cv2.rectangle(
            img,
            (x1, y1 - text_height - baseline - 5),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw label text
        cv2.putText(
            img,
            label_text,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
    
    return img


def visualize_predictions(
    image: np.ndarray,
    predictions: torch.Tensor,
    class_names: Optional[List[str]] = None,
    conf_threshold: float = 0.25,
) -> np.ndarray:
    """
    Visualize model predictions
    
    Args:
        image: Input image [H, W, 3]
        predictions: Predictions tensor [N, 6] (x, y, w, h, conf, cls)
        class_names: Optional list of class names
        conf_threshold: Confidence threshold
    
    Returns:
        Image with visualizations
    """
    # Filter by confidence
    mask = predictions[:, 4] > conf_threshold
    filtered_preds = predictions[mask]
    
    if len(filtered_preds) == 0:
        return image
    
    # Convert to (x1, y1, x2, y2) format
    boxes = []
    labels = []
    scores = []
    
    for pred in filtered_preds:
        x, y, w, h, conf, cls = pred
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)
        
        boxes.append((x1, y1, x2, y2))
        labels.append(int(cls))
        scores.append(float(conf))
    
    # Draw boxes
    img = draw_boxes(image, boxes, labels, scores, class_names)
    
    return img


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_metrics: Optional[dict] = None,
    val_metrics: Optional[dict] = None,
    save_path: Optional[str] = None,
):
    """
    Plot training curves
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        train_metrics: Optional dict of training metrics
        val_metrics: Optional dict of validation metrics
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Metrics curves
    if train_metrics and val_metrics:
        for metric_name in train_metrics.keys():
            if metric_name in val_metrics:
                axes[1].plot(train_metrics[metric_name], label=f'Train {metric_name}')
                axes[1].plot(val_metrics[metric_name], label=f'Val {metric_name}')
        
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Metric')
        axes[1].set_title('Training and Validation Metrics')
        axes[1].legend()
        axes[1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    else:
        plt.show()
    
    plt.close()

