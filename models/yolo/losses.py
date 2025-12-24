"""
YOLO Loss Functions
"""

import torch
import torch.nn as nn
from typing import List


# ============================
# CIoU LOSS
# ============================
class CIoULoss(nn.Module):
    """Complete IoU Loss"""

    def __init__(self):
        super().__init__()

    def forward(self, pred_boxes: torch.Tensor, target_boxes: torch.Tensor) -> torch.Tensor:
        # Convert (x, y, w, h) → (x1, y1, x2, y2)
        pred_x1 = pred_boxes[:, 0] - pred_boxes[:, 2] / 2
        pred_y1 = pred_boxes[:, 1] - pred_boxes[:, 3] / 2
        pred_x2 = pred_boxes[:, 0] + pred_boxes[:, 2] / 2
        pred_y2 = pred_boxes[:, 1] + pred_boxes[:, 3] / 2

        target_x1 = target_boxes[:, 0] - target_boxes[:, 2] / 2
        target_y1 = target_boxes[:, 1] - target_boxes[:, 3] / 2
        target_x2 = target_boxes[:, 0] + target_boxes[:, 2] / 2
        target_y2 = target_boxes[:, 1] + target_boxes[:, 3] / 2

        inter_x1 = torch.max(pred_x1, target_x1)
        inter_y1 = torch.max(pred_y1, target_y1)
        inter_x2 = torch.min(pred_x2, target_x2)
        inter_y2 = torch.min(pred_y2, target_y2)

        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        pred_area = pred_boxes[:, 2] * pred_boxes[:, 3]
        target_area = target_boxes[:, 2] * target_boxes[:, 3]

        union_area = pred_area + target_area - inter_area
        iou = inter_area / (union_area + 1e-7)

        center_dist = torch.sum((pred_boxes[:, :2] - target_boxes[:, :2]) ** 2, dim=1)

        enclose_x1 = torch.min(pred_x1, target_x1)
        enclose_y1 = torch.min(pred_y1, target_y1)
        enclose_x2 = torch.max(pred_x2, target_x2)
        enclose_y2 = torch.max(pred_y2, target_y2)

        enclose_diag = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7

        v = (4 / (torch.pi ** 2)) * (
            torch.atan(target_boxes[:, 2] / target_boxes[:, 3]) -
            torch.atan(pred_boxes[:, 2] / pred_boxes[:, 3])
        ) ** 2

        alpha = v / (1 - iou + v + 1e-7)

        ciou = iou - center_dist / enclose_diag - alpha * v
        return 1 - ciou.mean()


# ============================
# YOLO LOSS
# ============================
class YOLOLoss(nn.Module):
    """
    YOLO Loss compatible with targets [N, 6]:
    (batch_idx, class, x, y, w, h)
    """

    def __init__(
        self,
        num_classes: int,
        box_loss_weight: float = 7.5,
        cls_loss_weight: float = 0.5,
        obj_loss_weight: float = 1.0,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.box_loss_weight = box_loss_weight
        self.cls_loss_weight = cls_loss_weight
        self.obj_loss_weight = obj_loss_weight

        self.box_loss_fn = CIoULoss()
        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions: List[torch.Tensor], targets: torch.Tensor) -> dict:
        device = predictions[0].device

        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        if targets is None or targets.numel() == 0:
            return {
                "total_loss": total_box,
                "box_loss": total_box,
                "obj_loss": total_obj,
                "cls_loss": total_cls,
            }

        batch_idx = targets[:, 0].long()
        cls = targets[:, 1].long()
        boxes = targets[:, 2:6].clamp(0, 1)

        for pred in predictions:
            B, A, H, W, _ = pred.shape

            obj_mask = torch.zeros(B, A, H, W, device=device, dtype=torch.bool)
            tgt_boxes = torch.zeros(B, A, H, W, 4, device=device)
            tgt_cls = torch.zeros(B, A, H, W, self.num_classes, device=device)

            for i in range(targets.shape[0]):
                b = batch_idx[i]
                if b >= B:
                    continue

                x, y, w, h = boxes[i]
                gx, gy = int(x * W), int(y * H)
                gx = min(max(gx, 0), W - 1)
                gy = min(max(gy, 0), H - 1)

                obj_mask[b, 0, gy, gx] = True
                tgt_boxes[b, 0, gy, gx] = torch.tensor([x, y, w, h], device=device)
                tgt_cls[b, 0, gy, gx, cls[i]] = 1.0

            if obj_mask.any():
                total_box += self.box_loss_fn(pred[..., :4][obj_mask], tgt_boxes[obj_mask])
                total_cls += self.bce(pred[..., 5:][obj_mask], tgt_cls[obj_mask])

            total_obj += self.bce(pred[..., 4], obj_mask.float())

        total_loss = (
            self.box_loss_weight * total_box
            + self.obj_loss_weight * total_obj
            + self.cls_loss_weight * total_cls
        )

        return {
            "total_loss": total_loss,
            "box_loss": total_box,
            "obj_loss": total_obj,
            "cls_loss": total_cls,
        }
