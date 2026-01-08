#!/usr/bin/env python3
"""
Test batch size constraints and model variant detection.

Tests:
1. Text quantized model fails fast on batch_size > 1
2. Text FP32 model allows batching
3. Image models allow batching (both variants)
4. Batch size limits from env vars
"""

import sys
import subprocess
import time
import requests
import json
from pathlib import Path


def wait_for_server(url: str, max_wait: int = 30):
    """Wait for server to be ready."""
    for _ in range(max_wait):
        try:
            resp = requests.get(f"{url}/health", timeout=1)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def test_info_endpoint(url: str):
    """Test /info endpoint to check model variant detection."""
    print("\n" + "=" * 70)
    print("Testing /info endpoint (model variant detection)")
    print("=" * 70)
    
    resp = requests.get(f"{url}/info")
    if resp.status_code != 200:
        print(f"❌ Failed to get /info: {resp.status_code}")
        return False
    
    info = resp.json()
    print(f"\nModel info:")
    print(f"  Text model: {info.get('txt_model', 'N/A')}")
    print(f"  Text variant: {info.get('txt_model_variant', 'N/A')}")
    print(f"  Text max batch size: {info.get('txt_max_batch_size', 'N/A')}")
    print(f"  Image model: {info.get('img_model', 'N/A')}")
    print(f"  Image variant: {info.get('img_model_variant', 'N/A')}")
    print(f"  Image max batch size: {info.get('img_max_batch_size', 'N/A')}")
    
    return True


def test_text_batch_quantized(url: str):
    """Test text batch with quantized model (should fail if batch_size > 1)."""
    print("\n" + "=" * 70)
    print("Testing text batch with quantized model")
    print("=" * 70)
    
    # Get model variant from /info
    resp = requests.get(f"{url}/info")
    info = resp.json()
    variant = info.get('txt_model_variant', '')
    
    if variant != 'quantized':
        print(f"⚠️  Text model is {variant}, not quantized. Skipping quantized test.")
        return True
    
    # Test batch_size=1 (should work)
    print("\nTest 1: batch_size=1 (should work)")
    resp = requests.post(
        f"{url}/txt/batch",
        json={
            "inputs": ["Hello world"],
            "dim": 768,
            "prefix": "search_query"
        }
    )
    if resp.status_code == 200:
        print("  ✓ batch_size=1 works")
    else:
        print(f"  ✗ batch_size=1 failed: {resp.status_code} - {resp.text}")
        return False
    
    # Test batch_size=2 (should fail)
    print("\nTest 2: batch_size=2 (should fail fast)")
    resp = requests.post(
        f"{url}/txt/batch",
        json={
            "inputs": ["Hello world", "Goodbye world"],
            "dim": 768,
            "prefix": "search_query"
        }
    )
    if resp.status_code == 400:
        # Try to parse as JSON, fall back to text
        try:
            error_msg = resp.json().get('error', resp.text)
        except:
            error_msg = resp.text
        
        if 'batching is not supported' in error_msg.lower() or 'cross-sample interference' in error_msg.lower() or 'exceeds maximum allowed batch size' in error_msg.lower():
            print(f"  ✓ Correctly rejected batch_size=2")
            print(f"    Error: {error_msg[:150]}...")
            return True
        else:
            print(f"  ✗ Wrong error message: {error_msg[:200]}")
            return False
    else:
        print(f"  ✗ Expected 400, got {resp.status_code}: {resp.text[:200]}")
        return False


def test_text_batch_fp32(url: str):
    """Test text batch with FP32 model (should allow batching)."""
    print("\n" + "=" * 70)
    print("Testing text batch with FP32 model")
    print("=" * 70)
    
    # Get model variant from /info
    resp = requests.get(f"{url}/info")
    info = resp.json()
    variant = info.get('txt_model_variant', '')
    
    if variant != 'full':
        print(f"⚠️  Text model is {variant}, not full. Skipping FP32 test.")
        return True
    
    # Test batch_size=2 (should work)
    print("\nTest: batch_size=2 (should work)")
    resp = requests.post(
        f"{url}/txt/batch",
        json={
            "inputs": ["Hello world", "Goodbye world"],
            "dim": 768,
            "prefix": "search_query"
        }
    )
    if resp.status_code == 200:
        data = resp.json()
        if len(data.get('embeddings', [])) == 2:
            print("  ✓ batch_size=2 works")
            return True
        else:
            print(f"  ✗ Expected 2 embeddings, got {len(data.get('embeddings', []))}")
            return False
    else:
        print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
        return False


def test_image_batch(url: str):
    """Test image batch (should work for both variants)."""
    print("\n" + "=" * 70)
    print("Testing image batch")
    print("=" * 70)
    
    # Create a simple test image (1x1 red pixel, base64 encoded)
    import base64
    from PIL import Image
    import io
    
    img = Image.new('RGB', (224, 224), color='red')
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    img_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    
    # Test batch_size=2
    print("\nTest: batch_size=2 (should work)")
    resp = requests.post(
        f"{url}/img/batch",
        json={
            "contents": [f"data:image/png;base64,{img_base64}"] * 2,
            "dim": 768
        }
    )
    if resp.status_code == 200:
        data = resp.json()
        if len(data.get('embeddings', [])) == 2:
            print("  ✓ batch_size=2 works")
            return True
        else:
            print(f"  ✗ Expected 2 embeddings, got {len(data.get('embeddings', []))}")
            return False
    else:
        print(f"  ✗ Failed: {resp.status_code} - {resp.text}")
        return False


def test_batch_size_limit(url: str):
    """Test batch size limit enforcement."""
    print("\n" + "=" * 70)
    print("Testing batch size limits")
    print("=" * 70)
    
    # Get max batch sizes from /info
    resp = requests.get(f"{url}/info")
    info = resp.json()
    txt_max = info.get('txt_max_batch_size')
    img_max = info.get('img_max_batch_size')
    
    print(f"\nMax batch sizes:")
    print(f"  Text: {txt_max}")
    print(f"  Image: {img_max}")
    
    # Test text batch exceeding limit (if limit is set and < 100)
    if txt_max and txt_max < 100:
        print(f"\nTest: Text batch_size={txt_max + 1} (should fail)")
        resp = requests.post(
            f"{url}/txt/batch",
            json={
                "inputs": ["test"] * (txt_max + 1),
                "dim": 768,
                "prefix": "search_query"
            }
        )
        if resp.status_code == 400:
            print(f"  ✓ Correctly rejected batch_size={txt_max + 1}")
        else:
            print(f"  ✗ Expected 400, got {resp.status_code}")
            return False
    
    return True


def main():
    print("=" * 70)
    print("Batch Constraints Test")
    print("=" * 70)
    
    url = "http://localhost:8080"
    
    # Check if server is running
    if not wait_for_server(url):
        print(f"❌ Server not responding at {url}")
        print("   Start server with: cargo run --release")
        sys.exit(1)
    
    print(f"✓ Server is running at {url}")
    
    # Run tests
    results = []
    
    results.append(("Info endpoint", test_info_endpoint(url)))
    results.append(("Text batch (quantized)", test_text_batch_quantized(url)))
    results.append(("Text batch (FP32)", test_text_batch_fp32(url)))
    results.append(("Image batch", test_image_batch(url)))
    results.append(("Batch size limits", test_batch_size_limit(url)))
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✓" if passed else "✗"
        print(f"{status} {name}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("\n✅ All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Some tests failed")
        sys.exit(1)


if __name__ == "__main__":
    main()

