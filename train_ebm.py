"""
Training Script for Energy-Based Model on CIFAR-10
Assignment 4 - Part 1

Trains EBM using contrastive divergence.
Key: Demonstrates gradient descent on INPUT images for sampling.
"""

import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from ebm_model import EnergyBasedModel


def load_cifar10(batch_size=64, image_size=32):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='../sps_genai/data',
        train=True,
        download=False,
        transform=transform
    )
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='../sps_genai/data',
        train=False,
        download=False,
        transform=transform
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, test_loader


def train_ebm(model, train_loader, optimizer, epochs=30, device='cuda', 
              checkpoint_dir='models', num_langevin_steps=60):
    """
    Train Energy-Based Model using Contrastive Divergence
    
    Args:
        model: EnergyBasedModel instance
        train_loader: Training data loader
        optimizer: Optimizer
        epochs: Number of epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        num_langevin_steps: Steps for Langevin sampling
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    training_stats = []
    
    for epoch in range(epochs):
        model.train()
        epoch_losses = []
        epoch_real_energies = []
        epoch_fake_energies = []
        
        loader_with_progress = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs}')
        
        for batch_idx, (images, _) in enumerate(loader_with_progress):
            images = images.to(device)
            
            # Train step with contrastive divergence
            loss, real_energy, fake_energy = model.train_step_contrastive_divergence(
                real_images=images,
                optimizer=optimizer,
                num_langevin_steps=num_langevin_steps
            )
            
            epoch_losses.append(loss)
            epoch_real_energies.append(real_energy)
            epoch_fake_energies.append(fake_energy)
            
            loader_with_progress.set_postfix({
                'loss': f'{loss:.4f}',
                'E_real': f'{real_energy:.2f}',
                'E_fake': f'{fake_energy:.2f}'
            })
        
        # Epoch statistics
        avg_loss = sum(epoch_losses) / len(epoch_losses)
        avg_real_energy = sum(epoch_real_energies) / len(epoch_real_energies)
        avg_fake_energy = sum(epoch_fake_energies) / len(epoch_fake_energies)
        
        training_stats.append({
            'epoch': epoch + 1,
            'loss': avg_loss,
            'real_energy': avg_real_energy,
            'fake_energy': avg_fake_energy
        })
        
        print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | "
              f"E_real: {avg_real_energy:.2f} | E_fake: {avg_fake_energy:.2f}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
            'real_energy': avg_real_energy,
            'fake_energy': avg_fake_energy
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'ebm_epoch_{epoch+1:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model (maximize energy gap)
        energy_gap = avg_fake_energy - avg_real_energy
        if epoch == 0 or energy_gap > best_energy_gap:
            best_energy_gap = energy_gap
            best_checkpoint_path = os.path.join(checkpoint_dir, 'ebm_cifar10_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"✓ New best model saved (energy gap: {energy_gap:.2f})")
        
        # Generate samples every 5 epochs
        if (epoch + 1) % 5 == 0:
            print("Generating sample images...")
            sample_images = model.generate(num_images=16, num_steps=100)
            save_sample_grid(sample_images, epoch + 1, checkpoint_dir)
    
    return training_stats


def save_sample_grid(images, epoch, save_dir):
    """Save generated images in a grid"""
    import torchvision.utils as vutils
    
    grid = vutils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'EBM Generated Images - Epoch {epoch}')
    plt.savefig(os.path.join(save_dir, f'ebm_samples_epoch_{epoch:03d}.png'))
    plt.close()


def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING ENERGY-BASED MODEL ON CIFAR-10")
    print("=" * 60)
    print("\nKEY FEATURE: Gradient descent on INPUT images!")
    print("(Not on model parameters - that's the learning objective)")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 64
    IMAGE_SIZE = 32
    EPOCHS = 30
    LEARNING_RATE = 1e-4
    NUM_LANGEVIN_STEPS = 60
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Langevin steps: {NUM_LANGEVIN_STEPS}")
    print(f"  Device: {DEVICE}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, _ = load_cifar10(BATCH_SIZE, IMAGE_SIZE)
    print(f"✓ Train batches: {len(train_loader)}")
    
    # Create model
    print("\nCreating Energy-Based Model...")
    model = EnergyBasedModel(image_size=IMAGE_SIZE, num_channels=3)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, betas=(0.0, 0.999))
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    global best_energy_gap
    best_energy_gap = -float('inf')
    
    training_stats = train_ebm(
        model=model,
        train_loader=train_loader,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=DEVICE,
        checkpoint_dir='models',
        num_langevin_steps=NUM_LANGEVIN_STEPS
    )
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved at: models/ebm_cifar10_best.pth")
    print(f"Sample images saved in: models/")
    
    # Plot training curves
    epochs_list = [stat['epoch'] for stat in training_stats]
    losses = [stat['loss'] for stat in training_stats]
    real_energies = [stat['real_energy'] for stat in training_stats]
    fake_energies = [stat['fake_energy'] for stat in training_stats]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curve
    ax1.plot(epochs_list, losses, 'b-o')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('EBM Training Loss')
    ax1.grid(True)
    
    # Energy curves
    ax2.plot(epochs_list, real_energies, 'g-o', label='Real Images')
    ax2.plot(epochs_list, fake_energies, 'r-o', label='Generated Images')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Energy')
    ax2.set_title('Energy Values (Lower = More Realistic)')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('models/ebm_training_curves.png')
    print(f"Training curves saved at: models/ebm_training_curves.png")
    
    print("\n" + "=" * 60)
    print("KEY TAKEAWAY:")
    print("EBM learns an energy function, then uses gradient descent")
    print("on the INPUT images to find low-energy (realistic) states!")
    print("=" * 60)


if __name__ == "__main__":
    main()
