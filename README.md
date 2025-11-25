# Assignment 4: Advanced Image Generation

## Overview
Implementation of Diffusion Models and Energy-Based Models (EBMs) for image generation using the CIFAR-10 dataset, integrated with the Module 6 API.

## Assignment Components

### API Implementation
- **Diffusion Model**: Image generation using diffusion process on CIFAR-10
- **Energy-Based Model (EBM)**: Image generation via energy minimization with gradient descent on input
- **API Integration**: New endpoints added to Module 6 API

## Project Structure
```
assignment4_advanced_image_generation/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── diffusion_model.py          # Diffusion model implementation
├── ebm_model.py               # Energy-Based Model implementation
├── train_diffusion.py         # Training script for diffusion model
├── train_ebm.py               # Training script for EBM
├── app/
│   ├── main.py                # FastAPI with new endpoints
│   ├── diffusion_inference.py # Diffusion model inference
│   └── ebm_inference.py       # EBM inference
├── models/                    # Saved model weights
│   ├── diffusion_cifar10.pth
│   └── ebm_cifar10.pth
└── tests/
    ├── test_diffusion.py      # Diffusion model tests
    ├── test_ebm.py            # EBM tests
    └── test_api.py            # API endpoint tests
```


### Model Implementation
- **Diffusion**: Forward/reverse diffusion process, noise scheduling
- **EBM**: Energy function, Langevin dynamics, contrastive divergence

## Dataset
**CIFAR-10**: 60,000 32×32 color images in 10 classes
- 50,000 training images
- 10,000 test images
- Classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck

## Installation

```bash
pip install -r requirements.txt
```

## Training

### Train Diffusion Model
```bash
python train_diffusion.py --epochs 50 --batch-size 128
```

### Train Energy-Based Model
```bash
python train_ebm.py --epochs 30 --batch-size 64
```

## Running the API

### Local Development
```bash
uvicorn app.main:app --reload --port 8001
```

### Docker Deployment
```bash
docker build -t assignment4-image-gen .
docker run -d --name advanced-image-gen -p 8001:80 assignment4-image-gen
```

## API Endpoints

### New Endpoints (Assignment 4)
- `POST /generate/diffusion` - Generate images using diffusion model
- `POST /generate/ebm` - Generate images using energy-based model
- `GET /models/info` - Information about loaded models

### Existing Endpoints (From Module 6)
- `POST /generate` - Bigram text generation
- `POST /generate_with_rnn` - RNN text generation
- `POST /embed` - Word embeddings
- `GET /health` - Health check

## Testing

```bash
# Test diffusion model
python tests/test_diffusion.py

# Test EBM
python tests/test_ebm.py

# Test API
python tests/test_api.py
```


