import torch
import torch.nn as nn


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.channel_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # Channel attention
        ca = self.channel_att(x)
        x = x * ca

        # Spatial attention
        avg = torch.mean(x, dim=1, keepdim=True)
        max_, _ = torch.max(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([avg, max_], dim=1))

        return x * sa
