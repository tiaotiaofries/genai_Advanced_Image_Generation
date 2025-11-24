"""
Diffusion Model Inference Module
Loads trained diffusion model and generates images
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from diffusion_model import DiffusionModel


class DiffusionInference:
    """Inference wrapper for trained diffusion model"""
    
    def __init__(self, model_path: str, device: str = None):
        """
        Initialize diffusion inference
        
        Args:
            model_path: Path to trained model checkpoint
            device: Device to run inference on (cuda/cpu)
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.model_path = model_path
        self.is_loaded = False
        
    def load_model(self):
        """Load the trained diffusion model"""
        if self.is_loaded:
            return
            
        try:
            # Initialize model
            self.model = DiffusionModel(
                image_size=32,
                in_channels=3,
                embedding_dim=32,
                schedule_type='offset_cosine'
            ).to(self.device)
            
            # Load checkpoint
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # Load model weights (use EMA if available)
            if 'ema_model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['ema_model_state_dict'])
                print(f"Loaded EMA model from {self.model_path}")
            elif 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                print(f"Loaded model from {self.model_path}")
            else:
                self.model.load_state_dict(checkpoint)
                print(f"Loaded model weights from {self.model_path}")
            
            self.model.eval()
            self.is_loaded = True
            
            # Print model info
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"Diffusion model loaded: {total_params:,} parameters")
            
        except Exception as e:
            print(f"Error loading diffusion model: {e}")
            raise
    
    @torch.no_grad()
    def generate(
        self,
        num_images: int = 1,
        num_steps: int = 50,
        guidance_scale: float = 1.0,
        seed: int = None
    ) -> torch.Tensor:
        """
        Generate images using diffusion model
        
        Args:
            num_images: Number of images to generate
            num_steps: Number of denoising steps (less = faster, lower quality)
            guidance_scale: Guidance scale for generation (1.0 = no guidance)
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
        
        # Generate images
        images = self.model.generate(
            num_samples=num_images,
            num_steps=num_steps,
            device=self.device
        )
        
        return images
    
    def generate_with_class(
        self,
        class_idx: int,
        num_images: int = 1,
        num_steps: int = 50,
        seed: int = None
    ) -> torch.Tensor:
        """
        Generate images conditioned on class (if model supports it)
        
        Note: Current implementation is unconditional.
        This method is a placeholder for future conditional generation.
        
        Args:
            class_idx: CIFAR-10 class index (0-9)
            num_images: Number of images to generate
            num_steps: Number of denoising steps
            seed: Random seed
            
        Returns:
            Generated images
        """
        # For now, just generate unconditionally
        # TODO: Implement class-conditional generation
        print(f"Note: Class-conditional generation not yet implemented. Generating unconditionally.")
        return self.generate(num_images, num_steps, seed=seed)
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "status": "loaded",
            "model_type": "diffusion",
            "architecture": "UNet",
            "image_size": 32,
            "channels": 3,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
            "schedule_type": self.model.schedule_type,
            "timesteps": self.model.timesteps
        }


# Global instance (initialized in main.py)
diffusion_inference = None


def get_diffusion_inference() -> DiffusionInference:
    """Get the global diffusion inference instance"""
    global diffusion_inference
    if diffusion_inference is None:
        raise RuntimeError("Diffusion inference not initialized")
    return diffusion_inference
