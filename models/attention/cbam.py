"""
CBAM: Convolutional Block Attention Module
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Channel Attention Module"""
    
    def __init__(self, in_channels: int, reduction_ratio: int = 16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    """Spatial Attention Module"""
    
    def __init__(self, kernel_size: int = 7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        return self.sigmoid(out)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module
    Combines Channel and Spatial Attention
    """
    
    def __init__(
        self,
        in_channels: int,
        reduction_ratio: int = 16,
        kernel_size: int = 7,
        use_channel: bool = True,
        use_spatial: bool = True,
    ):
        super(CBAM, self).__init__()
        
        self.use_channel = use_channel
        self.use_spatial = use_spatial
        
        if use_channel:
            self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        
        if use_spatial:
            self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        if self.use_channel:
            x = x * self.channel_attention(x)
        
        if self.use_spatial:
            x = x * self.spatial_attention(x)
        
        return x

