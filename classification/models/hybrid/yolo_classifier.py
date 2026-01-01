import torch
import torch.nn as nn
from models.yolo.backbone import CSPDarknet
from .attention import CBAM


class YOLOClassifier(nn.Module):
    def __init__(self, num_classes=3, model_size="medium"):
        super().__init__()

        self.backbone = CSPDarknet(model_size=model_size)

        # Last feature map channels for YOLOv8-medium ≈ 768
        self.attention = CBAM(768)

        self.pool = nn.AdaptiveAvgPool2d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        features = self.backbone(x)
        x = features[-1]              # deepest feature
        x = self.attention(x)
        x = self.pool(x)
        return self.classifier(x)
