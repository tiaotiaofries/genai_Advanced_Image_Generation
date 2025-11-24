"""
Energy-Based Model (EBM) for CIFAR-10 Image Generation
Assignment 4 - Part 1

Key Learning Objective: Fine-grained gradient control
- Demonstrates gradient descent on INPUT IMAGES (not model parameters)
- Uses Langevin dynamics for sampling from learned energy function
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnergyFunction(nn.Module):
    """
    CNN-based energy function for EBM
    
    Maps images to scalar energy values.
    Lower energy = more realistic/likely images
    """
    
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        
        # Convolutional feature extractor
        self.features = nn.Sequential(
            # 32x32x3 -> 16x16x64
            nn.Conv2d(num_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(64),
            
            # 16x16x64 -> 8x8x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(128),
            
            # 8x8x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(256),
            
            # 4x4x256 -> 2x2x512
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.BatchNorm2d(512),
        )
        
        # Energy head - outputs single scalar
        self.energy_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 2 * 2, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        """
        Compute energy of input images
        
        Args:
            x: Input images [B, C, H, W]
        Returns:
            energy: Scalar energy values [B, 1]
        """
        features = self.features(x)
        energy = self.energy_head(features)
        return energy


class EnergyBasedModel(nn.Module):
    """
    Energy-Based Model for Image Generation
    
    Key Feature: Generates images via gradient descent on INPUT
    (not on model parameters!)
    
    This demonstrates fine-grained control over which tensors
    have gradients computed - a key learning objective of Assignment 4.
    """
    
    def __init__(self, image_size=32, num_channels=3):
        super().__init__()
        self.energy_fn = EnergyFunction(image_size, num_channels)
        self.image_size = image_size
        self.num_channels = num_channels
    
    def energy(self, x):
        """Compute energy of images"""
        return self.energy_fn(x)
    
    def sample_langevin(self, num_samples, num_steps=60, step_size=10.0, noise_scale=0.005):
        """
        Sample low-energy states using Langevin dynamics
        
        KEY ASSIGNMENT REQUIREMENT: Gradient descent on INPUT images
        
        Args:
            num_samples: Number of images to generate
            num_steps: Number of Langevin steps
            step_size: Step size for gradient descent
            noise_scale: Scale of noise added for exploration
            
        Returns:
            Generated images
        """
        device = next(self.parameters()).device
        
        # Initialize with random noise
        # KEY: requires_grad=True enables gradient computation for INPUT
        x = torch.randn(
            num_samples, self.num_channels, self.image_size, self.image_size,
            device=device,
            requires_grad=True  # ← CRITICAL: Gradients on INPUT, not params!
        )
        
        # Langevin dynamics: iteratively reduce energy
        for step in range(num_steps):
            # Compute energy
            energy = self.energy(x).sum()
            
            # KEY DEMONSTRATION: Gradient w.r.t. INPUT images
            # This is different from typical training where we do
            # gradients w.r.t. model parameters!
            energy.backward()
            
            with torch.no_grad():
                # Gradient descent on INPUT to minimize energy
                x.data -= step_size * x.grad
                
                # Add noise for exploration (Langevin dynamics)
                x.data += noise_scale * torch.randn_like(x)
                
                # Clip to valid range
                x.data.clamp_(-1, 1)
                
                # IMPORTANT: Zero gradients after update
                x.grad.zero_()
        
        # Detach from computation graph
        return x.detach()
    
    def train_step_contrastive_divergence(self, real_images, optimizer, num_langevin_steps=60):
        """
        Train using contrastive divergence
        
        Trains the energy function to assign:
        - Low energy to real images
        - High energy to generated (fake) images
        
        Args:
            real_images: Batch of real training images
            optimizer: Optimizer for energy function parameters
            num_langevin_steps: Steps for generating negative samples
            
        Returns:
            loss, real_energy_mean, fake_energy_mean
        """
        batch_size = real_images.size(0)
        
        # Energy of real images (should be low)
        real_energy = self.energy(real_images)
        
        # Generate negative samples using Langevin dynamics
        # KEY: Gradient descent on INPUT images
        with torch.no_grad():
            # Start from noise
            fake_images = self.sample_langevin(
                num_samples=batch_size,
                num_steps=num_langevin_steps
            )
        
        # Energy of fake images (should be high)
        fake_energy = self.energy(fake_images)
        
        # Contrastive divergence loss
        # Minimize energy of real, maximize energy of fake
        loss = real_energy.mean() - fake_energy.mean()
        
        # Backward pass (gradients w.r.t. energy function parameters)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss.item(), real_energy.mean().item(), fake_energy.mean().item()
    
    def generate(self, num_images, num_steps=100, step_size=10.0, noise_scale=0.005):
        """
        Generate images by finding low-energy states
        
        Uses Langevin dynamics with gradient descent on INPUT
        
        Args:
            num_images: Number of images to generate
            num_steps: Number of Langevin steps
            step_size: Step size for gradient descent
            noise_scale: Noise scale for exploration
            
        Returns:
            Generated images in [0, 1] range
        """
        self.eval()
        
        with torch.no_grad():
            # Use more steps for better generation quality
            images = self.sample_langevin(
                num_samples=num_images,
                num_steps=num_steps,
                step_size=step_size,
                noise_scale=noise_scale
            )
        
        # Convert from [-1, 1] to [0, 1]
        images = (images + 1) / 2
        images = torch.clamp(images, 0, 1)
        
        return images


def demonstrate_gradient_control():
    """
    Demonstration of fine-grained gradient control
    
    Shows the difference between:
    1. Gradients w.r.t. model parameters (normal training)
    2. Gradients w.r.t. input images (EBM sampling)
    
    This is the KEY LEARNING OBJECTIVE of Assignment 4!
    """
    print("=" * 60)
    print("DEMONSTRATING FINE-GRAINED GRADIENT CONTROL")
    print("=" * 60)
    
    # Create dummy EBM
    ebm = EnergyBasedModel(image_size=32, num_channels=3)
    
    print("\n1. NORMAL TRAINING: Gradients w.r.t. Model Parameters")
    print("-" * 60)
    
    # Dummy input (no gradients needed)
    x = torch.randn(4, 3, 32, 32, requires_grad=False)
    
    # Compute energy
    energy = ebm.energy(x).sum()
    energy.backward()
    
    # Check which tensors have gradients
    print("Input image has gradients:", x.requires_grad)
    for name, param in ebm.named_parameters():
        if param.grad is not None:
            print(f"✓ {name}: grad computed")
            break  # Just show first one
    
    print("\n2. EBM SAMPLING: Gradients w.r.t. INPUT Images")
    print("-" * 60)
    
    # Input with gradients enabled
    x = torch.randn(4, 3, 32, 32, requires_grad=True)  # ← KEY DIFFERENCE
    
    # Freeze model parameters
    for param in ebm.parameters():
        param.requires_grad = False
    
    # Compute energy
    energy = ebm.energy(x).sum()
    energy.backward()
    
    # Check gradients
    print("Input image has gradients:", x.requires_grad)
    print("Input gradient shape:", x.grad.shape if x.grad is not None else "None")
    print("Model parameters have gradients:", 
          any(p.requires_grad for p in ebm.parameters()))
    
    print("\n" + "=" * 60)
    print("KEY INSIGHT:")
    print("In EBM sampling, we do gradient descent on the INPUT")
    print("to find low-energy (realistic) images!")
    print("=" * 60)


if __name__ == "__main__":
    # Run demonstration
    demonstrate_gradient_control()
    
    # Test generation
    print("\n\nTesting EBM generation...")
    ebm = EnergyBasedModel()
    images = ebm.generate(num_images=4, num_steps=10)
    print(f"Generated {images.shape[0]} images of shape {images.shape[1:]}")
    print("✓ EBM implementation complete!")
