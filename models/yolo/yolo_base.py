"""
Baseline YOLO model implementation
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .backbone import CSPDarknet
from .head import YOLOHead
from .losses import YOLOLoss


class YOLOBase(nn.Module):
    """
    Baseline YOLO model for pomegranate growth stage classification
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_size: int = 640,
        model_size: str = "medium",
        anchors: Optional[List] = None,
    ):
        super(YOLOBase, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
        
        # Build backbone
        self.backbone = CSPDarknet(model_size=model_size)
        
        # Build neck (PANet)
        self.neck = self._build_neck()
        
        # Build head
        self.head = YOLOHead(
            num_classes=num_classes,
            anchors=anchors,
            model_size=model_size
        )
        
        # Loss function
        self.criterion = YOLOLoss(num_classes=num_classes)
        
    def _build_neck(self):
        """Build PANet neck"""
        # Get actual channel sizes from backbone based on model size
        size_configs = {
            "nano": {"width": 0.25, "depth": 0.33},
            "small": {"width": 0.50, "depth": 0.33},
            "medium": {"width": 0.75, "depth": 0.67},
            "large": {"width": 1.0, "depth": 1.0},
            "xlarge": {"width": 1.25, "depth": 1.33},
        }
        
        config = size_configs.get(self.model_size, size_configs["medium"])
        width_mult = config["width"]
        
        # Base channels from backbone stages
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_mult) for c in base_channels]
        
        # Neck channels: P3, P4, P5 correspond to channels[2], channels[3], channels[4]
        p3_channels = channels[2]  # 256 * 0.75 = 192 for medium
        p4_channels = channels[3]  # 512 * 0.75 = 384 for medium
        p5_channels = channels[4]  # 1024 * 0.75 = 768 for medium
        
        # Output channels for neck (reduced)
        neck_out_channels = [p3_channels // 2, p4_channels // 2, p5_channels // 2]
        
        # Simplified PANet implementation
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(p3_channels, neck_out_channels[0], 1),
                nn.BatchNorm2d(neck_out_channels[0]),
                nn.SiLU(),
                nn.Conv2d(neck_out_channels[0], neck_out_channels[0], 3, padding=1),
                nn.BatchNorm2d(neck_out_channels[0]),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(p4_channels, neck_out_channels[1], 1),
                nn.BatchNorm2d(neck_out_channels[1]),
                nn.SiLU(),
                nn.Conv2d(neck_out_channels[1], neck_out_channels[1], 3, padding=1),
                nn.BatchNorm2d(neck_out_channels[1]),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(p5_channels, neck_out_channels[2], 1),
                nn.BatchNorm2d(neck_out_channels[2]),
                nn.SiLU(),
                nn.Conv2d(neck_out_channels[2], neck_out_channels[2], 3, padding=1),
                nn.BatchNorm2d(neck_out_channels[2]),
                nn.SiLU(),
            ),
        ])
    
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

