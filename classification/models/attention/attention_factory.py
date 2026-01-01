"""
Attention Module Factory
"""

from typing import Optional
import torch
import torch.nn as nn
from .cbam import CBAM
from .se import SEBlock
from .coord_attention import CoordAttention


def create_attention(
    attention_type: str,
    in_channels: int,
    reduction_ratio: int = 16,
    kernel_size: int = 7,
    use_channel: bool = True,
    use_spatial: bool = True,
    **kwargs
) -> nn.Module:
    """
    Factory function to create attention modules
    
    Args:
        attention_type: Type of attention ('CBAM', 'SE', 'CoordAttention', 'SelfAttention')
        in_channels: Number of input channels
        reduction_ratio: Reduction ratio for channel attention
        kernel_size: Kernel size for spatial attention (CBAM)
        use_channel: Whether to use channel attention (CBAM)
        use_spatial: Whether to use spatial attention (CBAM)
        **kwargs: Additional arguments
    
    Returns:
        Attention module
    """
    if attention_type == "CBAM":
        return CBAM(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
            kernel_size=kernel_size,
            use_channel=use_channel,
            use_spatial=use_spatial,
        )
    elif attention_type == "SE":
        return SEBlock(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )
    elif attention_type == "CoordAttention":
        return CoordAttention(
            in_channels=in_channels,
            reduction_ratio=reduction_ratio,
        )
    elif attention_type == "SelfAttention":
        # Simple self-attention implementation
        return SelfAttention(
            in_channels=in_channels,
            num_heads=kwargs.get('num_heads', 8),
            dropout=kwargs.get('dropout', 0.1),
        )
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")


class SelfAttention(nn.Module):
    """
    Simple Self-Attention Module
    """
    
    def __init__(self, in_channels: int, num_heads: int = 8, dropout: float = 0.1):
        super(SelfAttention, self).__init__()
        self.num_heads = num_heads
        self.head_dim = in_channels // num_heads
        
        assert in_channels % num_heads == 0, "in_channels must be divisible by num_heads"
        
        self.query = nn.Conv2d(in_channels, in_channels, 1)
        self.key = nn.Conv2d(in_channels, in_channels, 1)
        self.value = nn.Conv2d(in_channels, in_channels, 1)
        
        self.out = nn.Conv2d(in_channels, in_channels, 1)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(in_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Attention-weighted tensor [B, C, H, W]
        """
        B, C, H, W = x.shape
        identity = x
        
        # Reshape for attention: [B, C, H, W] -> [B, C, H*W] -> [B, H*W, C]
        x_flat = x.view(B, C, H * W).permute(0, 2, 1)
        
        # Generate Q, K, V
        q = self.query(x).view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        k = self.key(x).view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        v = self.value(x).view(B, C, H * W).permute(0, 2, 1)  # [B, H*W, C]
        
        # Reshape for multi-head: [B, H*W, C] -> [B, H*W, num_heads, head_dim] -> [B, num_heads, H*W, head_dim]
        q = q.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B, H * W, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = torch.matmul(attn, v)  # [B, num_heads, H*W, head_dim]
        
        # Concatenate heads: [B, num_heads, H*W, head_dim] -> [B, H*W, C]
        out = out.permute(0, 2, 1, 3).contiguous().view(B, H * W, C)
        
        # Apply output projection
        out = self.out(out.permute(0, 2, 1).view(B, C, H, W))
        
        # Residual connection and normalization
        out = out + identity
        out = out.permute(0, 2, 3, 1)  # [B, H, W, C]
        out = self.norm(out)
        out = out.permute(0, 3, 1, 2)  # [B, C, H, W]
        
        return out

