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

        # ----------------------------
        # Backbone
        # ----------------------------
        self.backbone = CSPDarknet(model_size=model_size)

        # ----------------------------
        # Neck (PANet)
        # ----------------------------
        self.neck = self._build_neck()

        # ----------------------------
        # Head
        # ----------------------------
        self.head = YOLOHead(
            num_classes=num_classes,
            anchors=anchors,
            model_size=model_size
        )

        # ----------------------------
        # Loss
        # ----------------------------
        self.criterion = YOLOLoss(num_classes=num_classes)

    def _build_neck(self):
        """Build PANet neck"""

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

        p3_channels = channels[2]
        p4_channels = channels[3]
        p5_channels = channels[4]

        neck_out_channels = [
            p3_channels // 2,
            p4_channels // 2,
            p5_channels // 2,
        ]

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

    def forward(
        self,
        x: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
    ):
        """
        Forward pass

        Training:
            returns loss dict

        Inference:
            returns decoded predictions
        """

        # ----------------------------
        # Backbone
        # ----------------------------
        features = self.backbone(x)

        # ----------------------------
        # Neck
        # ----------------------------
        neck_features = []
        for i, neck_layer in enumerate(self.neck):
            neck_features.append(neck_layer(features[i]))

        # ----------------------------
        # Head
        # ----------------------------
        outputs = self.head(neck_features)

        # ----------------------------
        # TRAINING
        # ----------------------------
        if self.training and targets is not None:
            return self.criterion(outputs, targets)

        # ----------------------------
        # INFERENCE
        # ----------------------------
        return self.decode_predictions(outputs)

    def decode_predictions(self, outputs):
        """
        Decode raw YOLO outputs into:
        [x, y, w, h, conf, class_id]
        """

        decoded = []

        for out in outputs:
            # Expected shape: [B, A, H, W, 5 + C]
            out = torch.sigmoid(out)

            B, A, H, W, C = out.shape
            out = out.view(B, -1, C)

            # split
            box = out[..., :4]
            conf = out[..., 4:5]
            cls_scores = out[..., 5:]

            cls_conf, cls_id = cls_scores.max(dim=-1, keepdim=True)

            final_conf = conf * cls_conf
            det = torch.cat([box, final_conf, cls_id.float()], dim=-1)

            decoded.append(det)

        return torch.cat(decoded, dim=1)

    def predict(
        self,
        x: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
    ):
        """
        Inference helper with NMS
        """

        self.eval()
        with torch.no_grad():
            preds = self.forward(x)
            detections = self.head.nms(
                preds,
                conf_threshold=conf_threshold,
                iou_threshold=iou_threshold,
            )
        return detections
