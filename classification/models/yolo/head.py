"""
YOLO Detection Head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np


class YOLOHead(nn.Module):
    """
    YOLO detection head with classification and bounding box regression
    """
    
    def __init__(
        self,
        num_classes: int = 5,
        anchors: Optional[List] = None,
        model_size: str = "medium",
    ):
        super(YOLOHead, self).__init__()
        
        self.num_classes = num_classes
        self.model_size = model_size
        
        # Default anchors (for 640x640 input)
        if anchors is None:
            self.anchors = [
                [[10, 13], [16, 30], [33, 23]],  # P3
                [[30, 61], [62, 45], [59, 119]],  # P4
                [[116, 90], [156, 198], [373, 326]],  # P5
            ]
        else:
            self.anchors = anchors
        
        self.num_anchors = len(self.anchors[0])
        
        # Output channels: (num_anchors * (5 + num_classes))
        # 5 = 4 (bbox) + 1 (objectness)
        output_channels = self.num_anchors * (5 + num_classes)
        
        # Get actual channel sizes based on model size
        size_configs = {
            "nano": {"width": 0.25, "depth": 0.33},
            "small": {"width": 0.50, "depth": 0.33},
            "medium": {"width": 0.75, "depth": 0.67},
            "large": {"width": 1.0, "depth": 1.0},
            "xlarge": {"width": 1.25, "depth": 1.33},
        }
        
        config = size_configs.get(model_size, size_configs["medium"])
        width_mult = config["width"]
        
        # Base channels from backbone stages
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_mult) for c in base_channels]
        
        # Neck output channels (half of backbone channels)
        neck_channels = [channels[2] // 2, channels[3] // 2, channels[4] // 2]
        
        # Detection heads for each scale
        self.heads = nn.ModuleList([
            nn.Conv2d(neck_channels[0], output_channels, 1),  # P3
            nn.Conv2d(neck_channels[1], output_channels, 1),  # P4
            nn.Conv2d(neck_channels[2], output_channels, 1),  # P5
        ])
    
    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """
        Forward pass
        
        Args:
            features: List of feature maps from neck [P3, P4, P5]
        
        Returns:
            List of predictions at each scale
        """
        predictions = []
        for i, (feature, head) in enumerate(zip(features, self.heads)):
            pred = head(feature)
            B, C, H, W = pred.shape
            # Reshape: [B, num_anchors * (5 + num_classes), H, W] -> [B, num_anchors, 5 + num_classes, H, W]
            pred = pred.view(B, self.num_anchors, 5 + self.num_classes, H, W)
            pred = pred.permute(0, 1, 3, 4, 2).contiguous()  # [B, num_anchors, H, W, 5 + num_classes]
            predictions.append(pred)
        
        return predictions
    
    def decode_predictions(
        self,
        predictions: List[torch.Tensor],
        input_size: int = 640
    ) -> torch.Tensor:
        """
        Decode predictions to absolute coordinates
        
        Args:
            predictions: List of predictions from forward pass
            input_size: Input image size
        
        Returns:
            Decoded predictions [B, N, 6] (x, y, w, h, conf, cls)
        """
        device = predictions[0].device
        batch_size = predictions[0].shape[0]
        all_detections = []
        
        for scale_idx, pred in enumerate(predictions):
            B, A, H, W, C = pred.shape
            
            # Get anchors for this scale
            anchors = torch.tensor(self.anchors[scale_idx], device=device, dtype=torch.float32)
            
            # Create grid
            grid_y, grid_x = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            grid = torch.stack([grid_x, grid_y], dim=-1).float()  # [H, W, 2]
            
            # Decode bbox
            xy_pred = torch.sigmoid(pred[..., 0:2])  # [B, A, H, W, 2]
            wh_pred = pred[..., 2:4]  # [B, A, H, W, 2]
            obj_pred = torch.sigmoid(pred[..., 4:5])  # [B, A, H, W, 1]
            cls_pred = torch.sigmoid(pred[..., 5:])  # [B, A, H, W, num_classes]
            
            # Apply grid and anchors
            xy_pred = (xy_pred + grid.unsqueeze(0).unsqueeze(0)) * (input_size / max(H, W))
            wh_pred = torch.exp(wh_pred) * anchors.view(1, A, 1, 1, 2)
            
            # Get class scores
            cls_scores, cls_ids = torch.max(cls_pred, dim=-1, keepdim=True)
            conf = obj_pred * cls_scores
            
            # Reshape to [B, A*H*W, 6]
            detections = torch.cat([
                xy_pred[..., 0:1],  # x
                xy_pred[..., 1:2],  # y
                wh_pred[..., 0:1],  # w
                wh_pred[..., 1:2],  # h
                conf,  # confidence
                cls_ids.float(),  # class
            ], dim=-1)
            
            detections = detections.view(B, -1, 6)
            all_detections.append(detections)
        
        # Concatenate all scales
        all_detections = torch.cat(all_detections, dim=1)
        
        return all_detections
    
    @staticmethod
    def nms(
        predictions: torch.Tensor,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        max_detections: int = 1000
    ) -> List[torch.Tensor]:
        """
        Non-Maximum Suppression
        
        Args:
            predictions: [B, N, 6] (x, y, w, h, conf, cls)
            conf_threshold: Confidence threshold
            iou_threshold: IoU threshold for NMS
            max_detections: Maximum number of detections per image
        
        Returns:
            List of filtered detections per image
        """
        batch_size = predictions.shape[0]
        results = []
        
        for b in range(batch_size):
            dets = predictions[b]  # [N, 6]
            
            # Filter by confidence
            mask = dets[:, 4] > conf_threshold
            dets = dets[mask]
            
            if len(dets) == 0:
                results.append(torch.empty((0, 6), device=predictions.device))
                continue
            
            # Convert to (x1, y1, x2, y2) format
            boxes = dets[:, :4].clone()
            boxes[:, 0] = boxes[:, 0] - boxes[:, 2] / 2  # x1
            boxes[:, 1] = boxes[:, 1] - boxes[:, 3] / 2  # y1
            boxes[:, 2] = boxes[:, 0] + boxes[:, 2]  # x2
            boxes[:, 3] = boxes[:, 1] + boxes[:, 3]  # y2
            
            scores = dets[:, 4]
            classes = dets[:, 5]
            
            # Sort by confidence
            _, indices = torch.sort(scores, descending=True)
            boxes = boxes[indices]
            scores = scores[indices]
            classes = classes[indices]
            
            # NMS
            keep = []
            while len(indices) > 0:
                keep.append(indices[0].item())
                if len(indices) == 1:
                    break
                
                # Calculate IoU
                ious = YOLOHead._box_iou(boxes[0:1], boxes[1:])
                
                # Filter boxes with IoU > threshold
                mask = ious[0] < iou_threshold
                indices = indices[1:][mask]
                boxes = boxes[1:][mask]
                scores = scores[1:][mask]
                classes = classes[1:][mask]
            
            # Get kept detections
            if len(keep) > 0:
                kept_dets = dets[keep[:max_detections]]
                results.append(kept_dets)
            else:
                results.append(torch.empty((0, 6), device=predictions.device))
        
        return results
    
    @staticmethod
    def _box_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """
        Calculate IoU between boxes
        
        Args:
            box1: [N1, 4] (x1, y1, x2, y2)
            box2: [N2, 4] (x1, y1, x2, y2)
        
        Returns:
            [N1, N2] IoU matrix
        """
        # Calculate intersection
        inter_x1 = torch.max(box1[:, 0:1], box2[:, 0])
        inter_y1 = torch.max(box1[:, 1:2], box2[:, 1])
        inter_x2 = torch.min(box1[:, 2:3], box2[:, 2])
        inter_y2 = torch.min(box1[:, 3:4], box2[:, 3])
        
        inter_area = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)
        
        # Calculate union
        box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
        box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
        union_area = box1_area.unsqueeze(1) + box2_area.unsqueeze(0) - inter_area
        
        # IoU
        iou = inter_area / (union_area + 1e-7)
        
        return iou

