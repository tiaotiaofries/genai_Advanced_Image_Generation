#!/bin/bash

# Start API script for Assignment 4

echo "=========================================="
echo "Assignment 4: Advanced Image Generation"
echo "Starting FastAPI Server"
echo "=========================================="

# Check if models exist
if [ ! -f "models/diffusion_cifar10.pth" ]; then
    echo "⚠ Warning: models/diffusion_cifar10.pth not found"
    echo "  Train the diffusion model: python train_diffusion.py"
fi

if [ ! -f "models/ebm_cifar10.pth" ]; then
    echo "⚠ Warning: models/ebm_cifar10.pth not found"
    echo "  Train the EBM: python train_ebm.py"
fi

echo ""
echo "Starting server on http://localhost:8001"
echo "API docs at http://localhost:8001/docs"
echo ""
echo "Press Ctrl+C to stop"
echo "=========================================="

# Start uvicorn
uvicorn app.main:app --reload --port 8001
