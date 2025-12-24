"""
YOLO with Curriculum Learning Support
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .yolo_base import YOLOBase
from .losses import YOLOLoss


class YOLOCurriculum(nn.Module):
    """
    YOLO model with curriculum learning support
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_size: int = 640,
        model_size: str = "medium",
        anchors: Optional[List] = None,
        base_model: Optional[YOLOBase] = None,
    ):
        super(YOLOCurriculum, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
        
        # Use provided base model or create new one
        if base_model is not None:
            self.backbone = base_model.backbone
            self.neck = base_model.neck
            self.head = base_model.head
        else:
            from .yolo_base import YOLOBase
            base_model = YOLOBase(
                num_classes=num_classes,
                input_size=input_size,
                model_size=model_size,
                anchors=anchors
            )
            self.backbone = base_model.backbone
            self.neck = base_model.neck
            self.head = base_model.head
        
        # Loss function
        self.criterion = YOLOLoss(num_classes=num_classes)
        
        # Curriculum learning state
        self.current_stage = 1
        self.difficulty_threshold = 0.0
    
    def set_curriculum_stage(self, stage: int, difficulty_threshold: float = 0.0):
        """
        Set current curriculum learning stage
        
        Args:
            stage: Current stage (1, 2, or 3)
            difficulty_threshold: Difficulty threshold for filtering samples
        """
        self.current_stage = stage
        self.difficulty_threshold = difficulty_threshold
    
    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None):
        """
        Forward pass
        
        Args:
            x: Input tensor [B, 3, H, W]
            targets: Ground truth targets [B, N, 5] (class, x, y, w, h)
        
        Returns:
            If training: loss dict
            If inference: predictions
        """
        # Backbone
        features = self.backbone(x)
        
        # Neck
        neck_features = []
        for i, neck_layer in enumerate(self.neck):
            neck_features.append(neck_layer(features[i]))
        
        # Head
        outputs = self.head(neck_features)
        
        if self.training and targets is not None:
            # Calculate loss
            loss_dict = self.criterion(outputs, targets)
            return loss_dict
        else:
            # Return predictions
            return self.head.decode_predictions(outputs)
    
    def predict(self, x: torch.Tensor, conf_threshold: float = 0.25, iou_threshold: float = 0.45):
        """
        Prediction interface
        
        Args:
            x: Input tensor [B, 3, H, W]
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
        
        Returns:
            List of detections per image
        """
        self.eval()
        with torch.no_grad():
            predictions = self.forward(x)
            # Apply NMS
            detections = self.head.nms(predictions, conf_threshold, iou_threshold)
        return detections

