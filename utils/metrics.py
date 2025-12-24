"""
Evaluation metrics
"""

import numpy as np
import torch
from typing import List, Tuple, Optional
from collections import defaultdict


def calculate_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """
    Calculate IoU between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    if x2 <= x1 or y2 <= y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection
    
    return intersection / (union + 1e-7)


def convert_to_xyxy(box: np.ndarray) -> np.ndarray:
    """Convert (x_center, y_center, w, h) to (x1, y1, x2, y2)"""
    x, y, w, h = box
    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2
    return np.array([x1, y1, x2, y2])


def calculate_ap(
    predictions: List[np.ndarray],
    targets: List[np.ndarray],
    class_id: int,
    iou_threshold: float = 0.5,
) -> float:
    """
    Calculate Average Precision for a class
    
    Args:
        predictions: List of predictions [N, 6] (x, y, w, h, conf, cls)
        targets: List of targets [M, 5] (cls, x, y, w, h)
        class_id: Class ID to calculate AP for
        iou_threshold: IoU threshold
    
    Returns:
        AP value
    """
    # Filter predictions and targets by class
    pred_boxes = []
    pred_scores = []
    
    for pred in predictions:
        if isinstance(pred, torch.Tensor):
            pred = pred.cpu().numpy()
        
        for p in pred:
            if len(p) >= 6 and int(p[5]) == class_id:
                box = convert_to_xyxy(p[:4])
                pred_boxes.append(box)
                pred_scores.append(float(p[4]))
    
    target_boxes = []
    for target in targets:
        if isinstance(target, torch.Tensor):
            target = target.cpu().numpy()
        
        for t in target:
            if len(t) >= 5 and int(t[0]) == class_id:
                box = convert_to_xyxy(t[1:5])
                target_boxes.append(box)
    
    if len(pred_boxes) == 0:
        return 0.0 if len(target_boxes) > 0 else 1.0
    
    # Sort predictions by confidence
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = [pred_boxes[i] for i in sorted_indices]
    pred_scores = [pred_scores[i] for i in sorted_indices]
    
    # Match predictions to targets
    matched = [False] * len(target_boxes)
    tp = []
    fp = []
    
    for pred_box in pred_boxes:
        best_iou = 0.0
        best_idx = -1
        
        for i, target_box in enumerate(target_boxes):
            if matched[i]:
                continue
            
            iou = calculate_iou(pred_box, target_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        
        if best_iou >= iou_threshold:
            matched[best_idx] = True
            tp.append(1)
            fp.append(0)
        else:
            tp.append(0)
            fp.append(1)
    
    # Calculate precision and recall
    tp = np.cumsum(tp)
    fp = np.cumsum(fp)
    
    recalls = tp / (len(target_boxes) + 1e-7)
    precisions = tp / (tp + fp + 1e-7)
    
    # Calculate AP using 11-point interpolation
    ap = 0.0
    for t in np.arange(0, 1.1, 0.1):
        if np.sum(recalls >= t) == 0:
            p = 0
        else:
            p = np.max(precisions[recalls >= t])
        ap += p / 11.0
    
    return ap


def calculate_map(
    predictions: List,
    targets: List,
    iou_threshold: float = 0.5,
    map_range: Optional[Tuple[float, float]] = None,
    num_classes: int = 5,
) -> float:
    """
    Calculate mean Average Precision (mAP)
    
    Args:
        predictions: List of predictions
        targets: List of targets
        iou_threshold: IoU threshold (or start threshold if map_range provided)
        map_range: Optional (start, end) range for mAP calculation
        num_classes: Number of classes
    
    Returns:
        mAP value
    """
    if map_range is not None:
        # Calculate mAP@0.5:0.95
        aps = []
        for threshold in np.arange(map_range[0], map_range[1] + 0.05, 0.05):
            ap = calculate_map(predictions, targets, iou_threshold=threshold, num_classes=num_classes)
            aps.append(ap)
        return np.mean(aps)
    else:
        # Calculate mAP@iou_threshold
        aps = []
        for class_id in range(num_classes):
            ap = calculate_ap(predictions, targets, class_id, iou_threshold)
            aps.append(ap)
        return np.mean(aps)

