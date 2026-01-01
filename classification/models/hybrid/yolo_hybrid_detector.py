"""
Hybrid YOLO Detector
- Loads pretrained classifier backbone
- Freezes backbone
- Adds YOLO neck + head
"""

import torch
import torch.nn as nn

from models.yolo.backbone import CSPDarknet
from models.yolo.head import YOLOHead
from models.yolo.losses import YOLOLoss


class YOLOHybridDetector(nn.Module):
    def __init__(
        self,
        num_classes: int,
        classifier_ckpt: str,
        model_size: str = "medium",
        anchors=None,
    ):
        super().__init__()

        # -----------------------------
        # Backbone (classifier-trained)
        # -----------------------------
        self.backbone = CSPDarknet(model_size=model_size)

        state = torch.load(classifier_ckpt, map_location="cpu")
        backbone_state = {
            k.replace("backbone.", ""): v
            for k, v in state["model_state_dict"].items()
            if k.startswith("backbone.")
        }
        self.backbone.load_state_dict(backbone_state, strict=False)

        for p in self.backbone.parameters():
            p.requires_grad = False

        # -----------------------------
        # YOLO Neck (CRITICAL FIX)
        # -----------------------------
        self.neck = self._build_neck(model_size)

        # -----------------------------
        # YOLO Head
        # -----------------------------
        self.head = YOLOHead(
            num_classes=num_classes,
            anchors=anchors,
            model_size=model_size,
        )

        self.criterion = YOLOLoss(num_classes=num_classes)

    def _build_neck(self, model_size):
        size_cfg = {
            "nano": 0.25,
            "small": 0.50,
            "medium": 0.75,
            "large": 1.0,
        }
        w = size_cfg.get(model_size, 0.75)

        base = [256, 512, 1024]
        ch = [int(c * w) for c in base]

        return nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch[0], ch[0] // 2, 1),
                nn.BatchNorm2d(ch[0] // 2),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(ch[1], ch[1] // 2, 1),
                nn.BatchNorm2d(ch[1] // 2),
                nn.SiLU(),
            ),
            nn.Sequential(
                nn.Conv2d(ch[2], ch[2] // 2, 1),
                nn.BatchNorm2d(ch[2] // 2),
                nn.SiLU(),
            ),
        ])

    def forward(self, images, targets=None):
        features = self.backbone(images)

        neck_features = []
        for i, neck in enumerate(self.neck):
            neck_features.append(neck(features[i]))

        outputs = self.head(neck_features)

        if self.training:
            return self.criterion(outputs, targets)

        return self.head.decode_predictions(outputs)
