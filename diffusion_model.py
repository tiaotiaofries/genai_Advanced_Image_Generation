"""
Diffusion Model for CIFAR-10 Image Generation
Assignment 4 - Part 1

Based on Module 8 implementation, adapted for CIFAR-10 dataset.
Uses UNet architecture with diffusion process for image generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import copy


class SinusoidalEmbedding(nn.Module):
    """Sinusoidal positional embedding for noise variance encoding"""
    
    def __init__(self, num_frequencies=16):
        super().__init__()
        self.num_frequencies = num_frequencies
        frequencies = torch.exp(torch.linspace(math.log(1.0), math.log(1000.0), num_frequencies))
        self.register_buffer("angular_speeds", 2.0 * math.pi * frequencies.view(1, 1, 1, -1))

    def forward(self, x):
        """
        x: Tensor of shape (B, 1, 1, 1)
        returns: Tensor of shape (B, 1, 1, 2 * num_frequencies)
        """
        x = x.expand(-1, 1, 1, self.num_frequencies)
        sin_part = torch.sin(self.angular_speeds * x)
        cos_part = torch.cos(self.angular_speeds * x)
        return torch.cat([sin_part, cos_part], dim=-1)


def cosine_diffusion_schedule(diffusion_times):
    """
    Cosine diffusion schedule for smooth noise addition
    
    Args:
        diffusion_times: Tensor of values in [0, 1]
    Returns:
        noise_rates, signal_rates
    """
    signal_rates = torch.cos(diffusion_times * math.pi / 2)
    noise_rates = torch.sin(diffusion_times * math.pi / 2)
    return noise_rates, signal_rates


def offset_cosine_diffusion_schedule(diffusion_times, min_signal_rate=0.02, max_signal_rate=0.95):
    """
    Offset cosine schedule avoiding extreme values
    
    Args:
        diffusion_times: Tensor of values in [0, 1]
        min_signal_rate: Minimum signal rate
        max_signal_rate: Maximum signal rate
    Returns:
        noise_rates, signal_rates
    """
    original_shape = diffusion_times.shape
    diffusion_times_flat = diffusion_times.flatten()
    
    start_angle = torch.acos(torch.tensor(max_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    end_angle = torch.acos(torch.tensor(min_signal_rate, dtype=torch.float32, device=diffusion_times.device))
    
    diffusion_angles = start_angle + diffusion_times_flat * (end_angle - start_angle)
    
    signal_rates = torch.cos(diffusion_angles).reshape(original_shape)
    noise_rates = torch.sin(diffusion_angles).reshape(original_shape)
    
    return noise_rates, signal_rates


class ResidualBlock(nn.Module):
    """Residual block for UNet"""
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.needs_projection = in_channels != out_channels
        if self.needs_projection:
            self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.proj = nn.Identity()
        
        self.norm = nn.BatchNorm2d(in_channels, affine=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
    
    def swish(self, x):
        return x * torch.sigmoid(x)
    
    def forward(self, x):
        residual = self.proj(x)
        x = self.swish(self.conv1(x))
        x = self.conv2(x)
        return x + residual


class DownBlock(nn.Module):
    """Downsampling block for UNet"""
    
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels, width))
            in_channels = width
        self.pool = nn.AvgPool2d(kernel_size=2)
    
    def forward(self, x, skips):
        for block in self.blocks:
            x = block(x)
            skips.append(x)
        x = self.pool(x)
        return x


class UpBlock(nn.Module):
    """Upsampling block for UNet"""
    
    def __init__(self, width, block_depth, in_channels):
        super().__init__()
        self.blocks = nn.ModuleList()
        for _ in range(block_depth):
            self.blocks.append(ResidualBlock(in_channels + width, width))
            in_channels = width
    
    def forward(self, x, skips):
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
        for block in self.blocks:
            skip = skips.pop()
            x = torch.cat([x, skip], dim=1)
            x = block(x)
        return x


class UNet(nn.Module):
    """
    UNet architecture for diffusion model denoising
    
    Designed for CIFAR-10: 32x32x3 images
    """
    
    def __init__(self, image_size=32, num_channels=3, embedding_dim=32):
        super().__init__()
        self.initial = nn.Conv2d(num_channels, 32, kernel_size=1)
        self.num_channels = num_channels
        self.image_size = image_size
        self.embedding_dim = embedding_dim
        self.embedding = SinusoidalEmbedding(num_frequencies=16)
        self.embedding_proj = nn.Conv2d(embedding_dim, 32, kernel_size=1)
        
        self.down1 = DownBlock(32, in_channels=64, block_depth=2)
        self.down2 = DownBlock(64, in_channels=32, block_depth=2)
        self.down3 = DownBlock(96, in_channels=64, block_depth=2)
        
        self.mid1 = ResidualBlock(in_channels=96, out_channels=128)
        self.mid2 = ResidualBlock(in_channels=128, out_channels=128)
        
        self.up1 = UpBlock(96, in_channels=128, block_depth=2)
        self.up2 = UpBlock(64, block_depth=2, in_channels=96)
        self.up3 = UpBlock(32, block_depth=2, in_channels=64)
        
        self.final = nn.Conv2d(32, num_channels, kernel_size=1)
        nn.init.zeros_(self.final.weight)
    
    def forward(self, noisy_images, noise_variances):
        skips = []
        x = self.initial(noisy_images)
        noise_emb = self.embedding(noise_variances)
        noise_emb = F.interpolate(noise_emb.permute(0, 3, 1, 2), size=(self.image_size, self.image_size), mode='nearest')
        x = torch.cat([x, noise_emb], dim=1)
        
        x = self.down1(x, skips)
        x = self.down2(x, skips)
        x = self.down3(x, skips)
        
        x = self.mid1(x)
        x = self.mid2(x)
        
        x = self.up1(x, skips)
        x = self.up2(x, skips)
        x = self.up3(x, skips)
        
        return self.final(x)


class DiffusionModel(nn.Module):
    """
    Complete Diffusion Model for Image Generation
    
    Implements DDPM (Denoising Diffusion Probabilistic Model) for CIFAR-10
    """
    
    def __init__(self, image_size=32, num_channels=3, schedule_fn=offset_cosine_diffusion_schedule):
        super().__init__()
        self.network = UNet(image_size, num_channels)
        self.ema_network = copy.deepcopy(self.network)
        self.ema_network.eval()
        self.ema_decay = 0.999
        self.schedule_fn = schedule_fn
        self.normalizer_mean = 0.0
        self.normalizer_std = 1.0
    
    def to(self, device):
        super().to(device)
        self.ema_network.to(device)
        return self
    
    def set_normalizer(self, mean, std):
        """Set normalization parameters for denormalization"""
        self.normalizer_mean = mean
        self.normalizer_std = std
    
    def denormalize(self, x):
        """Denormalize images back to [0, 1] range"""
        return torch.clamp(x * self.normalizer_std + self.normalizer_mean, 0.0, 1.0)
    
    def denoise(self, noisy_images, noise_rates, signal_rates, training):
        """
        Denoise images using the network
        
        Args:
            noisy_images: Noisy images
            noise_rates: Noise rates from schedule
            signal_rates: Signal rates from schedule  
            training: Whether in training mode
        """
        if training:
            network = self.network
            network.train()
        else:
            network = self.ema_network
            network.eval()
        
        pred_noises = network(noisy_images, noise_rates ** 2)
        pred_images = (noisy_images - noise_rates * pred_noises) / signal_rates
        return pred_noises, pred_images
    
    def reverse_diffusion(self, initial_noise, diffusion_steps):
        """
        Generate images by reversing the diffusion process
        
        Args:
            initial_noise: Starting random noise
            diffusion_steps: Number of denoising steps
        Returns:
            Generated images
        """
        step_size = 1.0 / diffusion_steps
        current_images = initial_noise
        
        for step in range(diffusion_steps):
            t = torch.ones((initial_noise.shape[0], 1, 1, 1), device=initial_noise.device) * (1 - step * step_size)
            noise_rates, signal_rates = self.schedule_fn(t)
            pred_noises, pred_images = self.denoise(current_images, noise_rates, signal_rates, training=False)
            
            next_diffusion_times = t - step_size
            next_noise_rates, next_signal_rates = self.schedule_fn(next_diffusion_times)
            current_images = next_signal_rates * pred_images + next_noise_rates * pred_noises
        
        return pred_images
    
    def generate(self, num_images, diffusion_steps=20, image_size=32, initial_noise=None):
        """
        Generate new images from random noise
        
        Args:
            num_images: Number of images to generate
            diffusion_steps: Number of denoising steps (more = better quality)
            image_size: Size of images
            initial_noise: Optional initial noise (otherwise random)
        Returns:
            Generated images in [0, 1] range
        """
        if initial_noise is None:
            initial_noise = torch.randn(
                (num_images, self.network.num_channels, image_size, image_size),
                device=next(self.parameters()).device
            )
        
        with torch.no_grad():
            return self.denormalize(self.reverse_diffusion(initial_noise, diffusion_steps))
    
    def train_step(self, images, optimizer, loss_fn):
        """
        Single training step
        
        Args:
            images: Batch of training images
            optimizer: Optimizer
            loss_fn: Loss function (typically MSE)
        Returns:
            Loss value
        """
        # Normalize images
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        
        # Sample random diffusion times
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        
        # Get noise and signal rates
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        
        # Predict noise
        pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=True)
        
        # Compute loss
        loss = loss_fn(pred_noises, noises)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA network
        with torch.no_grad():
            for ema_param, param in zip(self.ema_network.parameters(), self.network.parameters()):
                ema_param.copy_(self.ema_decay * ema_param + (1. - self.ema_decay) * param)
        
        return loss.item()
    
    def test_step(self, images, loss_fn):
        """
        Single evaluation step
        
        Args:
            images: Batch of test images
            loss_fn: Loss function
        Returns:
            Loss value
        """
        images = (images - self.normalizer_mean) / self.normalizer_std
        noises = torch.randn_like(images)
        
        diffusion_times = torch.rand((images.size(0), 1, 1, 1), device=images.device)
        noise_rates, signal_rates = self.schedule_fn(diffusion_times)
        noisy_images = signal_rates * images + noise_rates * noises
        
        with torch.no_grad():
            pred_noises, _ = self.denoise(noisy_images, noise_rates, signal_rates, training=False)
            loss = loss_fn(pred_noises, noises)
        
        return loss.item()
