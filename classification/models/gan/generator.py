"""
FINAL FIXED Pix2Pix U-Net Generator
Channel-safe, skip-safe, Windows-safe
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DownBlock(nn.Module):
    def __init__(self, in_c, out_c, normalize=True):
        super().__init__()
        layers = [nn.Conv2d(in_c, out_c, 4, 2, 1, bias=not normalize)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_c))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UpBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetGenerator(nn.Module):
    def __init__(
        self,
        input_channels=3,
        output_channels=3,
        base_filters=64,
        num_downsampling=4,
        **kwargs
    ):
        super().__init__()

        # ---------------- Encoder ----------------
        self.downs = nn.ModuleList()
        in_c = input_channels

        for i in range(num_downsampling):
            out_c = base_filters * (2 ** i)
            self.downs.append(
                DownBlock(in_c, out_c, normalize=(i != 0))
            )
            in_c = out_c

        # ---------------- Bottleneck ----------------
        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_c, in_c * 2, 4, 2, 1),
            nn.ReLU(inplace=True)
        )

        # ---------------- Decoder ----------------
        self.ups = nn.ModuleList()
        in_c = in_c * 2  # bottleneck output

        for i in reversed(range(num_downsampling)):
            out_c = base_filters * (2 ** i)
            self.ups.append(
                UpBlock(in_c, out_c, dropout=(i < 2))
            )
            in_c = out_c * 2  # after skip concat

        # ---------------- Final ----------------
        # IMPORTANT: input channels = in_c (NOT in_c // 2)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(in_c, output_channels, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []

        # Encoder
        for down in self.downs:
            x = down(x)
            skips.append(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skips = skips[::-1]
        for up, skip in zip(self.ups, skips):
            x = up(x)

            # Spatial safety
            if x.shape[2:] != skip.shape[2:]:
                skip = F.interpolate(skip, size=x.shape[2:], mode="nearest")

            x = torch.cat([x, skip], dim=1)

        return self.final(x)
