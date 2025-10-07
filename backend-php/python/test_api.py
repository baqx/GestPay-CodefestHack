#!/usr/bin/env python3
"""
Test script for GestPay FacePay API
Tests face registration and verification endpoints
"""

import requests
import base64
import json
import os
from pathlib import Path

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_IMAGES_DIR = Path("test_images")

def load_test_image(image_path):
    """Load and encode test image as base64"""
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
            return base64.b64encode(image_data).decode('utf-8')
    except FileNotFoundError:
        print(f"Test image not found: {image_path}")
        return None

def test_health_check():
    """Test health check endpoint"""
    print("Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health")
        print(f"Status: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_face_registration(image_base64):
    """Test face registration endpoint"""
    print("\nTesting face registration...")
    
    payload = {
        "image_base64": image_base64
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/register-face", json=payload)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("success") and result.get("embedding"):
            print(f"✓ Face registered successfully! Embedding length: {len(result['embedding'])}")
            return result["embedding"]
        else:
            print(f"✗ Face registration failed: {result.get('message')}")
            return None
            
    except Exception as e:
        print(f"Registration test failed: {e}")
        return None

def test_face_verification(image_base64, stored_embeddings):
    """Test face verification endpoint"""
    print("\nTesting face verification...")
    
    payload = {
        "image_base64": image_base64,
        "stored_embeddings": stored_embeddings,
        "confidence_threshold": 0.6
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/verify-face", json=payload)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        
        if result.get("match"):
            print(f"✓ Face verified! User ID: {result.get('user_id')}, Confidence: {result.get('confidence')}")
        else:
            print(f"✗ Face verification failed: {result.get('message')}")
            
        return result
        
    except Exception as e:
        print(f"Verification test failed: {e}")
        return None

def create_sample_image_data():
    """Create a sample base64 image for testing (1x1 pixel PNG)"""
    # This is a minimal 1x1 pixel PNG image encoded as base64
    sample_png = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChAI9jU"
        "8t6gAAAABJRU5ErkJggg=="
    )
    return sample_png

def main():
    """Run all tests"""
    print("=" * 50)
    print("GestPay FacePay API Test Suite")
    print("=" * 50)
    
    # Test 1: Health check
    if not test_health_check():
        print("❌ API is not running. Please start the FastAPI server first.")
        print("Run: cd python && python main.py")
        return
    
    print("✅ API is running!")
    
    # Test 2: Face registration with sample data
    print("\n" + "=" * 30)
    sample_image = create_sample_image_data()
    
    if not sample_image:
        print("❌ No test image available")
        return
    
    # Register face
    embedding = test_face_registration(sample_image)
    
    if not embedding:
        print("❌ Face registration failed")
        return
    
    print("✅ Face registration successful!")
    
    # Test 3: Face verification
    print("\n" + "=" * 30)
    stored_embeddings = [
        {
            "user_id": 1,
            "embedding": embedding
        }
    ]
    
    verification_result = test_face_verification(sample_image, stored_embeddings)
    
    if verification_result and verification_result.get("match"):
        print("✅ Face verification successful!")
    else:
        print("❌ Face verification failed")
    
    # Test 4: Verification with different image (should fail)
    print("\n" + "=" * 30)
    print("Testing with different image (should fail)...")
    
    # Create a slightly different sample image
    different_sample = create_sample_image_data()  # Same for now, but in real test would be different
    
    verification_result2 = test_face_verification(different_sample, stored_embeddings)
    
    print("\n" + "=" * 50)
    print("Test Summary:")
    print("✅ Health Check: PASSED")
    print("✅ Face Registration: PASSED" if embedding else "❌ Face Registration: FAILED")
    print("✅ Face Verification (Same): PASSED" if verification_result and verification_result.get("match") else "❌ Face Verification (Same): FAILED")
    print("=" * 50)
    
    print("\n📝 Notes:")
    print("- This test uses a minimal sample image")
    print("- For real testing, use actual face photos")
    print("- Place test images in 'test_images/' directory")
    print("- Ensure the FastAPI server is running on localhost:8000")

if __name__ == "__main__":
    main()
