"""
CoordAttention: Coordinate Attention Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CoordAttention(nn.Module):
    """
    Coordinate Attention Module
    """
    
    def __init__(self, in_channels: int, reduction_ratio: int = 32):
        super(CoordAttention, self).__init__()
        
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        
        mip = max(8, in_channels // reduction_ratio)
        
        self.conv1 = nn.Conv2d(in_channels, mip, 1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = nn.ReLU(inplace=True)
        
        self.conv_h = nn.Conv2d(mip, in_channels, 1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, in_channels, 1, stride=1, padding=0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        identity = x
        
        n, c, h, w = x.size()
        
        # X-direction pooling
        x_h = self.pool_h(x)  # [B, C, H, 1]
        
        # Y-direction pooling
        x_w = self.pool_w(x).permute(0, 1, 3, 2)  # [B, C, 1, W] -> [B, C, W, 1]
        
        # Concatenate
        y = torch.cat([x_h, x_w], dim=2)  # [B, C, H+W, 1]
        
        # Shared MLP
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        
        # Split
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)  # [B, C, 1, W]
        
        # Generate attention maps
        a_h = self.conv_h(x_h).sigmoid()  # [B, C, H, 1]
        a_w = self.conv_w(x_w).sigmoid()  # [B, C, 1, W]
        
        # Apply attention
        out = identity * a_h * a_w
        
        return out

