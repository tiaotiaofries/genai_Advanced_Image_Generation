"""
API Endpoint Tests
Tests for the FastAPI image generation endpoints
"""

import requests
import json
import base64
from PIL import Image
import io
import numpy as np

# API base URL
BASE_URL = "http://localhost:8001"


def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("Testing: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    assert response.status_code == 200
    
    data = response.json()
    print(f"Status: {data['status']}")
    print(f"Models: {data['models']}")
    print(f"Device: {data['device']}")
    print("✓ Health check passed")


def test_models_info():
    """Test models info endpoint"""
    print("\n" + "="*60)
    print("Testing: Models Info")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/models/info")
    assert response.status_code == 200
    
    data = response.json()
    print("Diffusion Model:")
    if data['diffusion']:
        for key, value in data['diffusion'].items():
            print(f"  {key}: {value}")
    else:
        print("  Not loaded")
    
    print("\nEBM:")
    if data['ebm']:
        for key, value in data['ebm'].items():
            print(f"  {key}: {value}")
    else:
        print("  Not loaded")
    
    print("✓ Models info retrieved")


def test_diffusion_generation():
    """Test diffusion model generation"""
    print("\n" + "="*60)
    print("Testing: Diffusion Model Generation")
    print("="*60)
    
    payload = {
        "num_images": 2,
        "num_steps": 50,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/generate/diffusion", json=payload)
    
    if response.status_code == 503:
        print("⚠ Diffusion model not loaded - train it first")
        print("  Run: python train_diffusion.py")
        return
    
    assert response.status_code == 200
    
    data = response.json()
    print(f"Success: {data['success']}")
    print(f"Message: {data['message']}")
    print(f"Num Images: {data['num_images']}")
    print(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
    
    # Save images
    if data['images']:
        for i, img_base64 in enumerate(data['images']):
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(f"test_diffusion_{i}.png")
            print(f"✓ Saved test_diffusion_{i}.png ({img.size})")
    
    print("✓ Diffusion generation passed")


def test_ebm_generation():
    """Test EBM generation"""
    print("\n" + "="*60)
    print("Testing: Energy-Based Model Generation")
    print("="*60)
    
    payload = {
        "num_images": 2,
        "num_steps": 60,
        "step_size": 10.0,
        "noise_scale": 0.005,
        "seed": 42
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    
    response = requests.post(f"{BASE_URL}/generate/ebm", json=payload)
    
    if response.status_code == 503:
        print("⚠ EBM not loaded - train it first")
        print("  Run: python train_ebm.py")
        return
    
    assert response.status_code == 200
    
    data = response.json()
    print(f"Success: {data['success']}")
    print(f"Message: {data['message']}")
    print(f"Num Images: {data['num_images']}")
    print(f"Metadata: {json.dumps(data['metadata'], indent=2)}")
    print(f"Key Feature: {data['metadata']['key_feature']}")
    
    # Save images
    if data['images']:
        for i, img_base64 in enumerate(data['images']):
            img_bytes = base64.b64decode(img_base64)
            img = Image.open(io.BytesIO(img_bytes))
            img.save(f"test_ebm_{i}.png")
            print(f"✓ Saved test_ebm_{i}.png ({img.size})")
    
    print("✓ EBM generation passed")


def test_classes_endpoint():
    """Test CIFAR-10 classes endpoint"""
    print("\n" + "="*60)
    print("Testing: CIFAR-10 Classes")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/classes")
    assert response.status_code == 200
    
    data = response.json()
    print(f"Dataset: {data['dataset']}")
    print(f"Num Classes: {data['num_classes']}")
    print("Classes:")
    for idx, name in data['classes'].items():
        print(f"  {idx}: {name}")
    
    print("✓ Classes endpoint passed")


def run_all_tests():
    """Run all API tests"""
    print("\n" + "="*60)
    print("ASSIGNMENT 4: API TESTS")
    print("="*60)
    print("Make sure the API is running:")
    print("  uvicorn app.main:app --reload --port 8001")
    print("="*60)
    
    try:
        test_health()
        test_models_info()
        test_classes_endpoint()
        test_diffusion_generation()
        test_ebm_generation()
        
        print("\n" + "="*60)
        print("✓ ALL TESTS PASSED")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\n✗ ERROR: Cannot connect to API")
        print("  Start the API first: uvicorn app.main:app --reload --port 8001")
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}")
    except Exception as e:
        print(f"\n✗ ERROR: {e}")


if __name__ == "__main__":
    run_all_tests()
