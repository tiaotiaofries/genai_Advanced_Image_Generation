"""
Quick Demo Script
Demonstrates the API functionality without running the full server
"""

import torch
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# Add parent to path
sys.path.append(str(Path(__file__).parent))

from diffusion_model import DiffusionModel
from ebm_model import EnergyBasedModel, demonstrate_gradient_control


def demo_diffusion():
    """Demo diffusion model"""
    print("\n" + "="*60)
    print("DIFFUSION MODEL DEMO")
    print("="*60)
    
    # Create model
    model = DiffusionModel(
        image_size=32,
        in_channels=3,
        embedding_dim=32,
        schedule_type='offset_cosine'
    )
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Schedule: {model.schedule_type}")
    print(f"Timesteps: {model.timesteps}")
    
    # Generate sample images (random weights - won't look good)
    print("\nGenerating 4 sample images (with random weights)...")
    with torch.no_grad():
        images = model.generate(num_samples=4, num_steps=50, device=device)
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    plt.suptitle('Diffusion Model (Untrained)')
    plt.tight_layout()
    plt.savefig('demo_diffusion.png', dpi=150, bbox_inches='tight')
    print("✓ Saved demo_diffusion.png")
    
    print("\nNote: Images are random noise since model is untrained.")
    print("      Train with: python train_diffusion.py")


def demo_ebm():
    """Demo Energy-Based Model"""
    print("\n" + "="*60)
    print("ENERGY-BASED MODEL DEMO")
    print("="*60)
    
    # Create model
    model = EnergyBasedModel(image_size=32, in_channels=3)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    print(f"Device: {device}")
    print(f"Parameters: {sum(p.numel() for p in model.energy_fn.parameters()):,}")
    
    # Demonstrate gradient control
    print("\n" + "-"*60)
    print("DEMONSTRATING: Gradient Descent on INPUT Images")
    print("-"*60)
    demonstrate_gradient_control()
    
    # Generate sample images (random weights - won't look good)
    print("\nGenerating 4 sample images via Langevin dynamics...")
    with torch.no_grad():
        images = model.sample_langevin(
            num_samples=4,
            num_steps=60,
            device=device
        )
    
    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for i, ax in enumerate(axes):
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f'Sample {i+1}')
    plt.suptitle('Energy-Based Model (Untrained)')
    plt.tight_layout()
    plt.savefig('demo_ebm.png', dpi=150, bbox_inches='tight')
    print("✓ Saved demo_ebm.png")
    
    print("\nNote: Images are random since model is untrained.")
    print("      Train with: python train_ebm.py")


def main():
    """Run all demos"""
    print("\n" + "="*60)
    print("ASSIGNMENT 4: QUICK DEMO")
    print("Advanced Image Generation - Diffusion & EBM")
    print("="*60)
    
    demo_diffusion()
    demo_ebm()
    
    print("\n" + "="*60)
    print("DEMO COMPLETE")
    print("="*60)
    print("\nNext steps:")
    print("1. Train models:")
    print("   python train_diffusion.py")
    print("   python train_ebm.py")
    print("\n2. Start API:")
    print("   ./start_api.sh")
    print("   # or: uvicorn app.main:app --reload --port 8001")
    print("\n3. Test API:")
    print("   python tests/test_api.py")
    print("="*60)


if __name__ == "__main__":
    main()
