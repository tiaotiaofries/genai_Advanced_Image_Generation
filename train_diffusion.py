"""
Training Script for Diffusion Model on CIFAR-10
Assignment 4 - Part 1

Trains the diffusion model to generate CIFAR-10 images.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
from diffusion_model import DiffusionModel, offset_cosine_diffusion_schedule


def load_cifar10(batch_size=64, image_size=32):
    """Load CIFAR-10 dataset"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    train_dataset = torchvision.datasets.CIFAR10(
        root='../sps_genai/data',  # Use existing data
        train=True,
        download=False,  # Already downloaded
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


def train_diffusion(model, train_loader, val_loader, optimizer, loss_fn, 
                    epochs=50, device='cuda', checkpoint_dir='models'):
    """
    Train diffusion model
    
    Args:
        model: DiffusionModel instance
        train_loader: Training data loader
        val_loader: Validation data loader
        optimizer: Optimizer
        loss_fn: Loss function (MSE)
        epochs: Number of epochs
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    best_val_loss = float('inf')
    training_stats = []
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_losses = []
        
        loader_with_progress = tqdm(train_loader, ncols=120, desc=f'Epoch {epoch+1}/{epochs} [Train]')
        for images, _ in loader_with_progress:
            images = images.to(device)
            loss = model.train_step(images, optimizer, loss_fn)
            train_losses.append(loss)
            loader_with_progress.set_postfix(loss=f'{loss:.4f}')
        
        avg_train_loss = sum(train_losses) / len(train_losses)
        
        # Validation
        model.eval()
        val_losses = []
        for images, _ in tqdm(val_loader, desc=f"Epoch {epoch+1} [Val]", ncols=120):
            images = images.to(device)
            loss = model.test_step(images, loss_fn)
            val_losses.append(loss)
        
        avg_val_loss = sum(val_losses) / len(val_losses)
        
        # Save statistics
        training_stats.append({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss
        })
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'model_state_dict': model.network.state_dict(),
            'ema_model_state_dict': model.ema_network.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'normalizer_mean': model.normalizer_mean,
            'normalizer_std': model.normalizer_std
        }
        
        checkpoint_path = os.path.join(checkpoint_dir, f'diffusion_epoch_{epoch+1:03d}.pth')
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_checkpoint_path = os.path.join(checkpoint_dir, 'diffusion_cifar10_best.pth')
            torch.save(checkpoint, best_checkpoint_path)
            print(f"✓ New best model saved at epoch {epoch+1} with val_loss: {avg_val_loss:.4f}")
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # Generate sample images every 10 epochs
        if (epoch + 1) % 10 == 0:
            print("Generating sample images...")
            sample_images = model.generate(num_images=16, diffusion_steps=20, image_size=32)
            save_sample_grid(sample_images, epoch + 1, checkpoint_dir)
    
    return training_stats


def save_sample_grid(images, epoch, save_dir):
    """Save generated images in a grid"""
    import torchvision.utils as vutils
    
    grid = vutils.make_grid(images, nrow=4, normalize=True)
    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).cpu())
    plt.axis('off')
    plt.title(f'Generated Images - Epoch {epoch}')
    plt.savefig(os.path.join(save_dir, f'samples_epoch_{epoch:03d}.png'))
    plt.close()


def main():
    """Main training function"""
    print("=" * 60)
    print("TRAINING DIFFUSION MODEL ON CIFAR-10")
    print("=" * 60)
    
    # Configuration
    BATCH_SIZE = 64
    IMAGE_SIZE = 32
    EPOCHS = 50
    LEARNING_RATE = 2e-4
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Epochs: {EPOCHS}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Device: {DEVICE}")
    
    # Load data
    print("\nLoading CIFAR-10 dataset...")
    train_loader, val_loader = load_cifar10(BATCH_SIZE, IMAGE_SIZE)
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Create model
    print("\nCreating diffusion model...")
    model = DiffusionModel(
        image_size=IMAGE_SIZE,
        num_channels=3,
        schedule_fn=offset_cosine_diffusion_schedule
    )
    model.set_normalizer(mean=0.0, std=1.0)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"✓ Model created with {total_params:,} parameters")
    
    # Setup training
    optimizer = optim.Adam(model.network.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.MSELoss()
    
    # Train
    print("\nStarting training...")
    print("=" * 60)
    
    training_stats = train_diffusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=EPOCHS,
        device=DEVICE,
        checkpoint_dir='models'
    )
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print(f"\nBest model saved at: models/diffusion_cifar10_best.pth")
    print(f"Sample images saved in: models/")
    
    # Plot training curve
    epochs_list = [stat['epoch'] for stat in training_stats]
    train_losses = [stat['train_loss'] for stat in training_stats]
    val_losses = [stat['val_loss'] for stat in training_stats]
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs_list, train_losses, 'b-o', label='Training Loss')
    plt.plot(epochs_list, val_losses, 'r-o', label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Diffusion Model Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('models/training_curve.png')
    print(f"Training curve saved at: models/training_curve.png")


if __name__ == "__main__":
    main()
