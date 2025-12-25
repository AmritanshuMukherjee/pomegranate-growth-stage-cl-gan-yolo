import torch
import torch.nn as nn


class YOLOLoss(nn.Module):
    def __init__(
        self,
        num_classes,
        box_weight=7.5,
        obj_weight=1.0,
        cls_weight=0.5,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.cls_weight = cls_weight

        self.bce = nn.BCEWithLogitsLoss(reduction="mean")

    def forward(self, predictions, targets):
        """
        predictions: list of tensors [B, A, H, W, 5 + C]
        targets: [N, 6] → (batch_idx, class, x, y, w, h)
        """

        device = predictions[0].device
        total_box = torch.tensor(0.0, device=device)
        total_obj = torch.tensor(0.0, device=device)
        total_cls = torch.tensor(0.0, device=device)

        if targets.numel() == 0:
            return {
                "total_loss": total_box + total_obj + total_cls,
                "box_loss": total_box,
                "obj_loss": total_obj,
                "cls_loss": total_cls,
            }

        # 🔥 FIX: filter invalid class IDs
        targets = targets[targets[:, 1] < self.num_classes]

        if targets.numel() == 0:
            return {
                "total_loss": total_box + total_obj + total_cls,
                "box_loss": total_box,
                "obj_loss": total_obj,
                "cls_loss": total_cls,
            }

        for pred in predictions:
            B, A, H, W, _ = pred.shape

            obj_target = torch.zeros((B, A, H, W), device=device)
            cls_target = torch.zeros((B, A, H, W, self.num_classes), device=device)

            for t in targets:
                b, cls, x, y, w, h = t
                b = int(b)
                cls = int(cls)

                gx = min(int(x * W), W - 1)
                gy = min(int(y * H), H - 1)

                obj_target[b, 0, gy, gx] = 1.0
                cls_target[b, 0, gy, gx, cls] = 1.0

            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            total_obj += self.bce(pred_obj, obj_target)
            total_cls += self.bce(pred_cls, cls_target)

        total_loss = (
            self.box_weight * total_box
            + self.obj_weight * total_obj
            + self.cls_weight * total_cls
        )

        return {
            "total_loss": total_loss,
            "box_loss": total_box,
            "obj_loss": total_obj,
            "cls_loss": total_cls,
        }
