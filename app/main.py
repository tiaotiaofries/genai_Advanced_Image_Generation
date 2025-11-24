"""
Assignment 4: Advanced Image Generation API
FastAPI server with Diffusion and Energy-Based Model endpoints
"""

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, List
import torch
import io
from PIL import Image
import numpy as np
from pathlib import Path

# Import inference modules
from app.diffusion_inference import DiffusionInference, diffusion_inference, get_diffusion_inference
from app.ebm_inference import EBMInference, ebm_inference, get_ebm_inference
import app.diffusion_inference as diff_module
import app.ebm_inference as ebm_module

# Initialize FastAPI
app = FastAPI(
    title="Assignment 4: Advanced Image Generation",
    description="Diffusion Models and Energy-Based Models for CIFAR-10",
    version="1.0.0"
)

# CIFAR-10 class names
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Model paths
MODELS_DIR = Path(__file__).parent.parent / "models"
DIFFUSION_MODEL_PATH = MODELS_DIR / "diffusion_cifar10.pth"
EBM_MODEL_PATH = MODELS_DIR / "ebm_cifar10.pth"


# ==================== Request/Response Models ====================

class DiffusionGenerateRequest(BaseModel):
    """Request model for diffusion generation"""
    num_images: int = Field(1, ge=1, le=16, description="Number of images to generate (1-16)")
    num_steps: int = Field(50, ge=10, le=1000, description="Number of denoising steps (10-1000)")
    guidance_scale: float = Field(1.0, ge=0.0, le=10.0, description="Guidance scale (not used in unconditional)")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    return_format: str = Field("base64", description="Return format: 'base64' or 'png'")


class EBMGenerateRequest(BaseModel):
    """Request model for EBM generation"""
    num_images: int = Field(1, ge=1, le=16, description="Number of images to generate (1-16)")
    num_steps: int = Field(60, ge=10, le=200, description="Number of Langevin steps (10-200)")
    step_size: float = Field(10.0, ge=0.1, le=50.0, description="Langevin step size")
    noise_scale: float = Field(0.005, ge=0.0, le=1.0, description="Noise scale for sampling")
    seed: Optional[int] = Field(None, description="Random seed for reproducibility")
    return_format: str = Field("base64", description="Return format: 'base64' or 'png'")


class GenerateResponse(BaseModel):
    """Response model for image generation"""
    success: bool
    message: str
    num_images: int
    images: Optional[List[str]] = None  # Base64 encoded images
    metadata: dict


# ==================== Startup/Shutdown ====================

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    print("=" * 60)
    print("Starting Assignment 4: Advanced Image Generation API")
    print("=" * 60)
    
    # Initialize diffusion model
    try:
        if DIFFUSION_MODEL_PATH.exists():
            diff_module.diffusion_inference = DiffusionInference(str(DIFFUSION_MODEL_PATH))
            diff_module.diffusion_inference.load_model()
            print("✓ Diffusion model loaded successfully")
        else:
            print(f"⚠ Diffusion model not found at {DIFFUSION_MODEL_PATH}")
            print("  Train the model first: python train_diffusion.py")
    except Exception as e:
        print(f"✗ Error loading diffusion model: {e}")
    
    # Initialize EBM
    try:
        if EBM_MODEL_PATH.exists():
            ebm_module.ebm_inference = EBMInference(str(EBM_MODEL_PATH))
            ebm_module.ebm_inference.load_model()
            print("✓ EBM loaded successfully")
        else:
            print(f"⚠ EBM not found at {EBM_MODEL_PATH}")
            print("  Train the model first: python train_ebm.py")
    except Exception as e:
        print(f"✗ Error loading EBM: {e}")
    
    print("=" * 60)
    print("API ready! Documentation at http://localhost:8001/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down API...")


# ==================== Health Check ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Assignment 4: Advanced Image Generation API",
        "endpoints": {
            "docs": "/docs",
            "health": "/health",
            "diffusion": "/generate/diffusion",
            "ebm": "/generate/ebm",
            "models_info": "/models/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    diffusion_status = "loaded" if diff_module.diffusion_inference and diff_module.diffusion_inference.is_loaded else "not_loaded"
    ebm_status = "loaded" if ebm_module.ebm_inference and ebm_module.ebm_inference.is_loaded else "not_loaded"
    
    return {
        "status": "healthy",
        "models": {
            "diffusion": diffusion_status,
            "ebm": ebm_status
        },
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }


# ==================== Model Info ====================

@app.get("/models/info")
async def models_info():
    """Get information about loaded models"""
    info = {
        "diffusion": None,
        "ebm": None
    }
    
    try:
        if diff_module.diffusion_inference:
            info["diffusion"] = diff_module.diffusion_inference.get_model_info()
    except:
        pass
    
    try:
        if ebm_module.ebm_inference:
            info["ebm"] = ebm_module.ebm_inference.get_model_info()
    except:
        pass
    
    return info


# ==================== Image Generation Endpoints ====================

def tensor_to_base64(images: torch.Tensor) -> List[str]:
    """Convert tensor images to base64 strings"""
    images_list = []
    
    # Convert to numpy and scale to [0, 255]
    images_np = (images.cpu().numpy() * 255).astype(np.uint8)
    
    for img_np in images_np:
        # Transpose from (C, H, W) to (H, W, C)
        img_np = np.transpose(img_np, (1, 2, 0))
        
        # Convert to PIL Image
        img_pil = Image.fromarray(img_np)
        
        # Convert to base64
        buffer = io.BytesIO()
        img_pil.save(buffer, format='PNG')
        import base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        images_list.append(img_base64)
    
    return images_list


@app.post("/generate/diffusion", response_model=GenerateResponse)
async def generate_diffusion(request: DiffusionGenerateRequest):
    """
    Generate images using Diffusion Model
    
    The diffusion model generates images by iteratively denoising random noise.
    This implements the reverse diffusion process learned during training.
    """
    try:
        # Check if model is loaded
        if not diff_module.diffusion_inference or not diff_module.diffusion_inference.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="Diffusion model not loaded. Please train the model first."
            )
        
        # Generate images
        images = diff_module.diffusion_inference.generate(
            num_images=request.num_images,
            num_steps=request.num_steps,
            guidance_scale=request.guidance_scale,
            seed=request.seed
        )
        
        # Convert to base64
        images_base64 = tensor_to_base64(images)
        
        return GenerateResponse(
            success=True,
            message=f"Generated {request.num_images} images using diffusion model",
            num_images=request.num_images,
            images=images_base64,
            metadata={
                "model": "diffusion",
                "num_steps": request.num_steps,
                "guidance_scale": request.guidance_scale,
                "seed": request.seed,
                "image_size": "32x32",
                "format": "RGB"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


@app.post("/generate/ebm", response_model=GenerateResponse)
async def generate_ebm(request: EBMGenerateRequest):
    """
    Generate images using Energy-Based Model (EBM)
    
    The EBM generates images via Langevin dynamics - iteratively moving random
    noise to lower energy states using GRADIENT DESCENT ON INPUT IMAGES.
    
    This demonstrates the key learning objective: fine-grained gradient control!
    """
    try:
        # Check if model is loaded
        if not ebm_module.ebm_inference or not ebm_module.ebm_inference.is_loaded:
            raise HTTPException(
                status_code=503,
                detail="EBM not loaded. Please train the model first."
            )
        
        # Generate images via Langevin dynamics
        # This uses gradient descent on INPUT images, not model parameters!
        images = ebm_module.ebm_inference.generate(
            num_images=request.num_images,
            num_steps=request.num_steps,
            step_size=request.step_size,
            noise_scale=request.noise_scale,
            seed=request.seed
        )
        
        # Convert to base64
        images_base64 = tensor_to_base64(images)
        
        return GenerateResponse(
            success=True,
            message=f"Generated {request.num_images} images using EBM (Langevin dynamics)",
            num_images=request.num_images,
            images=images_base64,
            metadata={
                "model": "energy_based_model",
                "num_steps": request.num_steps,
                "step_size": request.step_size,
                "noise_scale": request.noise_scale,
                "seed": request.seed,
                "sampling_method": "langevin_dynamics",
                "key_feature": "gradient_descent_on_input_images",
                "image_size": "32x32",
                "format": "RGB"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating images: {str(e)}")


# ==================== Additional Endpoints ====================

@app.get("/classes")
async def get_classes():
    """Get CIFAR-10 class names"""
    return {
        "dataset": "CIFAR-10",
        "num_classes": 10,
        "classes": {i: name for i, name in enumerate(CIFAR10_CLASSES)}
    }


@app.post("/compute_energy")
async def compute_energy(image_base64: str):
    """
    Compute energy of an image using the EBM
    
    Lower energy = more likely to be real data
    """
    try:
        if not ebm_module.ebm_inference or not ebm_module.ebm_inference.is_loaded:
            raise HTTPException(status_code=503, detail="EBM not loaded")
        
        # Decode base64 image
        import base64
        image_bytes = base64.b64decode(image_base64)
        image_pil = Image.open(io.BytesIO(image_bytes))
        
        # Convert to tensor
        image_np = np.array(image_pil).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)
        
        # Compute energy
        energy = ebm_module.ebm_inference.compute_energy(image_tensor)
        
        return {
            "energy": float(energy.item()),
            "message": "Lower energy indicates higher likelihood of being real data"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error computing energy: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
