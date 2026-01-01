"""
PatchGAN Discriminator for Pix2Pix
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator
    """
    
    def __init__(
        self,
        input_channels: int = 6,  # 3 (image) + 3 (generated)
        base_filters: int = 64,
        num_layers: int = 3,
        use_batch_norm: bool = True,
    ):
        super(PatchGANDiscriminator, self).__init__()
        
        self.num_layers = num_layers
        
        # First layer (no batch norm)
        layers = [
            nn.Conv2d(input_channels, base_filters, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        
        # Intermediate layers
        for i in range(num_layers - 1):
            in_channels = base_filters * (2 ** i)
            out_channels = base_filters * (2 ** (i + 1))
            
            layers.append(
                nn.Conv2d(in_channels, out_channels, 4, stride=2, padding=1, bias=False)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Final layer
        final_channels = base_filters * (2 ** (num_layers - 1))
        layers.append(
            nn.Conv2d(final_channels, 1, 4, stride=1, padding=1)
        )
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, image: torch.Tensor, generated: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image: Real image [B, 3, H, W]
            generated: Generated image [B, 3, H, W]
        
        Returns:
            Discriminator output [B, 1, H', W']
        """
        # Concatenate real and generated images
        x = torch.cat([image, generated], dim=1)  # [B, 6, H, W]
        
        # Forward through discriminator
        output = self.model(x)
        
        return output

