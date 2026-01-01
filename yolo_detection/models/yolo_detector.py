import torch
import torch.nn as nn

from models.yolo.backbone import CSPDarknet
from models.yolo.head import YOLOHead
from models.yolo.losses import YOLOLoss


class YOLODetector(nn.Module):
    """
    Single-stage YOLO Detector
    (Detection + Classification together)
    """

    def __init__(self, num_classes, model_size="medium", anchors=None):
        super().__init__()

        self.backbone = CSPDarknet(model_size=model_size)
        self.head = YOLOHead(
            num_classes=num_classes,
            anchors=anchors,
            model_size=model_size,
        )

        self.criterion = YOLOLoss(num_classes=num_classes)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        outputs = self.head(features)

        if self.training:
            return self.criterion(outputs, targets)

        return self.head.decode_predictions(outputs)
