"""
GAN Trainer for Pix2Pix
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, Optional, Tuple
from .generator import UNetGenerator
from .discriminator import PatchGANDiscriminator


class GANTrainer:
    """
    Trainer for Pix2Pix GAN
    """
    
    def __init__(
        self,
        generator: UNetGenerator,
        discriminator: PatchGANDiscriminator,
        device: str = "cuda",
        generator_lr: float = 0.0002,
        discriminator_lr: float = 0.0002,
        beta1: float = 0.5,
        beta2: float = 0.999,
        gan_loss_weight: float = 1.0,
        l1_loss_weight: float = 100.0,
    ):
        self.generator = generator.to(device)
        self.discriminator = discriminator.to(device)
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=generator_lr,
            betas=(beta1, beta2)
        )
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=discriminator_lr,
            betas=(beta1, beta2)
        )
        
        # Loss functions
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()
        
        self.gan_loss_weight = gan_loss_weight
        self.l1_loss_weight = l1_loss_weight
    
    def train_step(
        self,
        real_images: torch.Tensor,
        target_images: torch.Tensor
    ) -> Dict[str, float]:
        """
        Single training step
        
        Args:
            real_images: Real input images [B, 3, H, W]
            target_images: Target images [B, 3, H, W]
        
        Returns:
            Dictionary of losses
        """
        real_images = real_images.to(self.device)
        target_images = target_images.to(self.device)
        
        # Create fake images
        fake_images = self.generator(real_images)
        
        # ========== Train Discriminator ==========
        self.optimizer_d.zero_grad()
        
        # Real pair
        pred_real = self.discriminator(real_images, target_images)
        loss_d_real = self.criterion_gan(pred_real, torch.ones_like(pred_real))
        
        # Fake pair
        pred_fake = self.discriminator(real_images, fake_images.detach())
        loss_d_fake = self.criterion_gan(pred_fake, torch.zeros_like(pred_fake))
        
        # Discriminator loss
        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d.backward()
        self.optimizer_d.step()
        
        # ========== Train Generator ==========
        self.optimizer_g.zero_grad()
        
        # GAN loss
        pred_fake = self.discriminator(real_images, fake_images)
        loss_g_gan = self.criterion_gan(pred_fake, torch.ones_like(pred_fake))
        
        # L1 loss
        loss_g_l1 = self.criterion_l1(fake_images, target_images)
        
        # Generator loss
        loss_g = self.gan_loss_weight * loss_g_gan + self.l1_loss_weight * loss_g_l1
        loss_g.backward()
        self.optimizer_g.step()
        
        return {
            'loss_d': loss_d.item(),
            'loss_d_real': loss_d_real.item(),
            'loss_d_fake': loss_d_fake.item(),
            'loss_g': loss_g.item(),
            'loss_g_gan': loss_g_gan.item(),
            'loss_g_l1': loss_g_l1.item(),
        }
    
    def generate(self, real_images: torch.Tensor) -> torch.Tensor:
        """
        Generate images
        
        Args:
            real_images: Input images [B, 3, H, W]
        
        Returns:
            Generated images [B, 3, H, W]
        """
        self.generator.eval()
        with torch.no_grad():
            fake_images = self.generator(real_images.to(self.device))
        self.generator.train()
        return fake_images
    
    def save_checkpoint(self, path: str):
        """Save checkpoint"""
        torch.save({
            'generator_state_dict': self.generator.state_dict(),
            'discriminator_state_dict': self.discriminator.state_dict(),
            'optimizer_g_state_dict': self.optimizer_g.state_dict(),
            'optimizer_d_state_dict': self.optimizer_d.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load checkpoint"""
        checkpoint = torch.load(path, map_location=self.device)
        self.generator.load_state_dict(checkpoint['generator_state_dict'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        self.optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
        self.optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])

