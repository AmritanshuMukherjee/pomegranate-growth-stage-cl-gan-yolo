"""
YOLO Backbone: CSPDarknet
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class ConvBNSiLU(nn.Module):
    """Convolution + BatchNorm + SiLU activation"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super(ConvBNSiLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C3Block(nn.Module):
    """C3 block with bottleneck structure"""
    
    def __init__(self, in_channels: int, out_channels: int, num_blocks: int = 1, shortcut: bool = True):
        super(C3Block, self).__init__()
        hidden_channels = out_channels // 2
        
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv3 = ConvBNSiLU(2 * hidden_channels, out_channels, 1)
        
        self.bottlenecks = nn.ModuleList([
            nn.Sequential(
                ConvBNSiLU(hidden_channels, hidden_channels, 1),
                ConvBNSiLU(hidden_channels, hidden_channels, 3, padding=1)
            ) for _ in range(num_blocks)
        ])
        self.shortcut = shortcut and num_blocks > 0
    
    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        
        for bottleneck in self.bottlenecks:
            if self.shortcut:
                x1 = x1 + bottleneck(x1)
            else:
                x1 = bottleneck(x1)
        
        out = torch.cat([x1, x2], dim=1)
        return self.conv3(out)


class SPPF(nn.Module):
    """Spatial Pyramid Pooling - Fast"""
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 5):
        super(SPPF, self).__init__()
        hidden_channels = in_channels // 2
        self.conv1 = ConvBNSiLU(in_channels, hidden_channels, 1)
        self.conv2 = ConvBNSiLU(hidden_channels * 4, out_channels, 1)
        self.maxpool = nn.MaxPool2d(kernel_size, stride=1, padding=kernel_size // 2)
    
    def forward(self, x):
        x = self.conv1(x)
        x1 = self.maxpool(x)
        x2 = self.maxpool(x1)
        x3 = self.maxpool(x2)
        return self.conv2(torch.cat([x, x1, x2, x3], dim=1))


class CSPDarknet(nn.Module):
    """
    CSPDarknet backbone for YOLO
    """
    
    def __init__(self, model_size: str = "medium"):
        super(CSPDarknet, self).__init__()
        
        # Model size configurations
        size_configs = {
            "nano": {"width": 0.25, "depth": 0.33},
            "small": {"width": 0.50, "depth": 0.33},
            "medium": {"width": 0.75, "depth": 0.67},
            "large": {"width": 1.0, "depth": 1.0},
            "xlarge": {"width": 1.25, "depth": 1.33},
        }
        
        config = size_configs.get(model_size, size_configs["medium"])
        width_mult = config["width"]
        depth_mult = config["depth"]
        
        # Base channels
        base_channels = [64, 128, 256, 512, 1024]
        channels = [int(c * width_mult) for c in base_channels]
        
        # Depth multipliers
        depths = [int(3 * depth_mult), int(6 * depth_mult), int(6 * depth_mult), int(3 * depth_mult)]
        
        # Stem
        self.stem = ConvBNSiLU(3, channels[0], 6, stride=2, padding=2)
        
        # Stage 1
        self.stage1 = nn.Sequential(
            ConvBNSiLU(channels[0], channels[1], 3, stride=2, padding=1),
            C3Block(channels[1], channels[1], depths[0])
        )
        
        # Stage 2
        self.stage2 = nn.Sequential(
            ConvBNSiLU(channels[1], channels[2], 3, stride=2, padding=1),
            C3Block(channels[2], channels[2], depths[1])
        )
        
        # Stage 3
        self.stage3 = nn.Sequential(
            ConvBNSiLU(channels[2], channels[3], 3, stride=2, padding=1),
            C3Block(channels[3], channels[3], depths[2])
        )
        
        # Stage 4
        self.stage4 = nn.Sequential(
            ConvBNSiLU(channels[3], channels[4], 3, stride=2, padding=1),
            C3Block(channels[4], channels[4], depths[3]),
            SPPF(channels[4], channels[4])
        )
    
    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Forward pass
        
        Returns:
            List of feature maps at different scales
        """
        x = self.stem(x)
        x = self.stage1(x)
        p3 = self.stage2(x)  # 1/8 scale
        p4 = self.stage3(p3)  # 1/16 scale
        p5 = self.stage4(p4)  # 1/32 scale
        
        return [p3, p4, p5]

