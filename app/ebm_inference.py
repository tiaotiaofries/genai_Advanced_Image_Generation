"""
Energy-Based Model Inference Module
Loads trained EBM and generates images via Langevin dynamics
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from ebm_model import EnergyBasedModel


class EBMInference:
    """Inference wrapper for trained Energy-Based Model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize EBM inference
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained EBM"""
        if self.is_loaded:
            return
            
        try:
            # Initialize model
            self.model = EnergyBasedModel(
                image_size=32,
                in_channels=3
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load model weights
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded EBM from {self.model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded EBM weights from {self.model_path}")
            
            self.model.eval()
            self.is_loaded = True
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.energy_fn.parameters())
            print(f"EBM loaded: {total_params:,} parameters")
            
        except Exception as e:
            print(f"Error loading EBM: {e}")
            raise
    
    @torch.no_grad()
    def generate(
        self,
        num_images: int = 1,
        num_steps: int = 60,
        step_size: float = 10.0,
        noise_scale: float = 0.005,
        seed: int = None
    ) -> torch.Tensor:
        """
        Generate images using Langevin dynamics
        
        This demonstrates GRADIENT DESCENT ON INPUT IMAGES!
        The key learning objective of this assignment.
        
        Args:
            num_images: Number of images to generate
            num_steps: Number of Langevin steps
            step_size: Step size for gradient descent
            noise_scale: Noise scale for sampling
            seed: Random seed for reproducibility
            
        Returns:
            Generated images as tensor [N, 3, 32, 32] in range [0, 1]
        """
        if not self.is_loaded:
            self.load_model()
        
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        # Generate images via Langevin dynamics
        # Note: This uses gradient descent on INPUT images, not model parameters!
        images = self.model.sample_langevin(
            num_samples=num_images,
            num_steps=num_steps,
            step_size=step_size,
            noise_scale=noise_scale,
            device=self.device
        )
        
        return images
    
    def compute_energy(self, images: torch.Tensor) -> torch.Tensor:
        """
        Compute energy of given images
        
        Lower energy = more likely to be real data
        
        Args:
            images: Input images [N, 3, 32, 32]
            
        Returns:
            Energy values [N]
        """
        if not self.is_loaded:
            self.load_model()
        
        with torch.no_grad():
            images = images.to(self.device)
            energies = self.model.energy_fn(images).squeeze()
        
        return energies
    
    def refine_image(
        self,
        image: torch.Tensor,
        num_steps: int = 30,
        step_size: float = 5.0,
        noise_scale: float = 0.003
    ) -> torch.Tensor:
        """
        Refine an existing image to lower energy state
        
        Demonstrates gradient descent on a specific input image
        
        Args:
            image: Input image [3, 32, 32] or [1, 3, 32, 32]
            num_steps: Number of refinement steps
            step_size: Step size for gradient descent
            noise_scale: Noise scale
            
        Returns:
            Refined image
        """
        if not self.is_loaded:
            self.load_model()
        
        # Ensure batch dimension
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        # Start from the given image
        x = image.clone()
        
        # Refine via Langevin dynamics
        for step in range(num_steps):
            x.requires_grad = True
            
            # Compute energy and gradient w.r.t. INPUT image
            energy = self.model.energy_fn(x).sum()
            energy.backward()
            
            # Update image (gradient DESCENT to minimize energy)
            with torch.no_grad():
                x = x - step_size * x.grad
                
                # Add noise for exploration
                if step < num_steps - 1:
                    x = x + noise_scale * torch.randn_like(x)
                
                # Clamp to valid range
                x = torch.clamp(x, 0, 1)
            
            x = x.detach()
        
        return x
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.energy_fn.parameters())
        trainable_params = sum(p.numel() for p in self.model.energy_fn.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_type": "energy_based_model",
            "architecture": "CNN Energy Function",
            "image_size": 32,
            "channels": 3,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "sampling_method": "langevin_dynamics",
            "key_feature": "gradient_descent_on_input_images"
        }


# Global instance (initialized in main.py)
ebm_inference = None


def get_ebm_inference() -> EBMInference:
    """Get the global EBM inference instance"""
    global ebm_inference
    if ebm_inference is None:
        raise RuntimeError("EBM inference not initialized")
    return ebm_inference
