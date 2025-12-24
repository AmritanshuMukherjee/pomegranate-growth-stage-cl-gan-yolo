"""
YOLO with Attention Mechanisms
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
from .backbone import CSPDarknet, C3Block, SPPF, ConvBNSiLU
from .head import YOLOHead
from .losses import YOLOLoss
from ..attention.attention_factory import create_attention


class YOLOAttention(nn.Module):
    """
    YOLO model with attention mechanisms integrated
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        input_size: int = 640,
        model_size: str = "medium",
        anchors: Optional[List] = None,
        attention_type: str = "CBAM",
        attention_config: Optional[Dict] = None,
        attention_placement: Optional[List[str]] = None,
    ):
        super(YOLOAttention, self).__init__()
        
        self.num_classes = num_classes
        self.input_size = input_size
        self.model_size = model_size
        
        # Default attention placement
        if attention_placement is None:
            attention_placement = ["C3_1", "C3_2", "C3_3", "SPPF"]
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                "reduction_ratio": 16,
                "kernel_size": 7,
                "use_channel": True,
                "use_spatial": True,
            }
        
        # Build backbone with attention
        self.backbone = self._build_backbone_with_attention(
            attention_type=attention_type,
            attention_config=attention_config,
            attention_placement=attention_placement
        )
        
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
    
    def _build_backbone_with_attention(
        self,
        attention_type: str,
        attention_config: Dict,
        attention_placement: List[str]
    ):
        """Build CSPDarknet backbone with attention modules"""
        from .backbone import CSPDarknet
        
        # Get base backbone
        base_backbone = CSPDarknet(model_size=self.model_size)
        
        # Model size configurations
        size_configs = {
            "nano": {"width": 0.25, "depth": 0.33},
            "small": {"width": 0.50, "depth": 0.33},
            "medium": {"width": 0.75, "depth": 0.67},
            "large": {"width": 1.0, "depth": 1.0},
            "xlarge": {"width": 1.25, "depth": 1.33},
        }
        
        config = size_configs.get(self.model_size, size_configs["medium"])
        width_mult = config["width"]
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_mult) for c in base_channels]
        
        # Create new backbone with attention
        backbone = nn.ModuleDict()
        
        # Stem
        backbone['stem'] = base_backbone.stem
        
        # Stage 1
        backbone['stage1'] = base_backbone.stage1
        if "C3_1" in attention_placement:
            # Add attention after C3 block in stage1
            backbone['stage1_attention'] = create_attention(
                attention_type=attention_type,
                in_channels=channels[1],
                **attention_config
            )
        
        # Stage 2
        backbone['stage2'] = base_backbone.stage2
        if "C3_2" in attention_placement:
            backbone['stage2_attention'] = create_attention(
                attention_type=attention_type,
                in_channels=channels[2],
                **attention_config
            )
        
        # Stage 3
        backbone['stage3'] = base_backbone.stage3
        if "C3_3" in attention_placement:
            backbone['stage3_attention'] = create_attention(
                attention_type=attention_type,
                in_channels=channels[3],
                **attention_config
            )
        
        # Stage 4
        backbone['stage4'] = base_backbone.stage4
        if "SPPF" in attention_placement:
            # Add attention after SPPF
            backbone['stage4_attention'] = create_attention(
                attention_type=attention_type,
                in_channels=channels[4],
                **attention_config
            )
        
        return backbone
    
    def _build_neck(self):
        """Build PANet neck"""
        # Simplified PANet implementation
        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
                nn.Conv2d(128, 128, 3, padding=1),
                nn.BatchNorm2d(128),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
                nn.Conv2d(256, 256, 3, padding=1),
                nn.BatchNorm2d(256),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.BatchNorm2d(512),
                nn.SiLU(),
                nn.Conv2d(512, 512, 3, padding=1),
                nn.BatchNorm2d(512),
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
        # Backbone with attention
        x = self.backbone['stem'](x)
        
        x = self.backbone['stage1'](x)
        if 'stage1_attention' in self.backbone:
            x = self.backbone['stage1_attention'](x)
        
        p3 = self.backbone['stage2'](x)
        if 'stage2_attention' in self.backbone:
            p3 = self.backbone['stage2_attention'](p3)
        
        p4 = self.backbone['stage3'](p3)
        if 'stage3_attention' in self.backbone:
            p4 = self.backbone['stage3_attention'](p4)
        
        p5 = self.backbone['stage4'](p4)
        if 'stage4_attention' in self.backbone:
            p5 = self.backbone['stage4_attention'](p5)
        
        features = [p3, p4, p5]
        
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

