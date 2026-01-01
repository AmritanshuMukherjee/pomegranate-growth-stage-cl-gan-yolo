import torch
import torch.nn as nn

# -------------------------------------------------
# IMPORTANT: absolute imports from PROJECT ROOT
# -------------------------------------------------
from models.yolo.backbone import CSPDarknet
from models.yolo.head import YOLOHead
from models.yolo.losses import YOLOLoss


class YOLODetector(nn.Module):
    """
    Pure YOLO Detection Model
    (No classification head, no hybrid logic)
    """

    def __init__(
        self,
        num_classes: int,
        model_size: str = "medium",
        anchors=None,
    ):
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

        if self.training and targets is not None:
            return self.criterion(outputs, targets)

        return self.head.decode_predictions(outputs)
