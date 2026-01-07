#!/usr/bin/env python3
"""
Test Rust implementation of vision batching via API.

This tests the actual API endpoint to verify batching works correctly.
"""

import sys
import time
import requests
import json
from pathlib import Path

def test_batch_endpoint(base_url: str = "http://localhost:8080"):
    """Test the /img/batch endpoint with various scenarios."""
    
    print("=" * 70)
    print("Testing Rust Vision Batching Implementation")
    print("=" * 70)
    print(f"\nServer: {base_url}")
    print("Make sure server is running: make run")
    
    # Test images (using URLs for simplicity)
    test_images = [
        "https://picsum.photos/400/300",
        "https://picsum.photos/300/400",
        "https://picsum.photos/200/200",
        "https://picsum.photos/500/300",
    ]
    
    print(f"\n{'=' * 70}")
    print("TEST 1: Batch of 2 different images")
    print(f"{'=' * 70}")
    
    # Single inference for reference
    single_1 = requests.post(
        f"{base_url}/img/embed",
        json={"content": test_images[0]},
        timeout=30
    ).json()
    
    single_2 = requests.post(
        f"{base_url}/img/embed",
        json={"content": test_images[1]},
        timeout=30
    ).json()
    
    # Batch inference
    batch_response = requests.post(
        f"{base_url}/img/batch",
        json={"contents": test_images[:2]},
        timeout=30
    ).json()
    
    batch_embs = batch_response["embeddings"]
    
    # Compare
    import numpy as np
    
    def cosine_sim(a, b):
        return np.dot(a, b)
    
    def max_diff(a, b):
        return np.abs(np.array(a) - np.array(b)).max()
    
    diff_1 = max_diff(single_1["embedding"], batch_embs[0])
    diff_2 = max_diff(single_2["embedding"], batch_embs[1])
    cos_1 = cosine_sim(single_1["embedding"], batch_embs[0])
    cos_2 = cosine_sim(single_2["embedding"], batch_embs[1])
    
    print(f"Image 1: max_diff={diff_1:.6f}, cos_sim={cos_1:.6f}")
    print(f"Image 2: max_diff={diff_2:.6f}, cos_sim={cos_2:.6f}")
    
    # Check if FP32 or quantized (based on similarity)
    is_perfect = cos_1 > 0.9999 and cos_2 > 0.9999
    is_acceptable = cos_1 > 0.98 and cos_2 > 0.98
    
    if is_perfect:
        print("✅ Perfect batching (likely FP32 model)")
    elif is_acceptable:
        print("✅ Acceptable batching (quantized model, ~1% difference)")
    else:
        print("❌ Batching shows significant interference")
        return False
    
    print(f"\n{'=' * 70}")
    print("TEST 2: Batch of 4 different images")
    print(f"{'=' * 70}")
    
    # Single inference for all
    single_embs = []
    for img in test_images:
        resp = requests.post(
            f"{base_url}/img/embed",
            json={"content": img},
            timeout=30
        ).json()
        single_embs.append(resp["embedding"])
    
    # Batch inference
    batch_response = requests.post(
        f"{base_url}/img/batch",
        json={"contents": test_images},
        timeout=30
    ).json()
    
    batch_embs = batch_response["embeddings"]
    
    print(f"Batch size: {len(batch_embs)}")
    print(f"Processing time: {batch_response['time_ms']:.2f}ms")
    
    # Compare all
    max_diffs = []
    cos_sims = []
    for i, (single, batch) in enumerate(zip(single_embs, batch_embs)):
        diff = max_diff(single, batch)
        cos = cosine_sim(single, batch)
        max_diffs.append(diff)
        cos_sims.append(cos)
        print(f"  Image {i+1}: max_diff={diff:.6f}, cos_sim={cos:.6f}")
    
    avg_cos = np.mean(cos_sims)
    if avg_cos > 0.9999:
        print(f"\n✅ Perfect batching across all 4 images (avg cos_sim: {avg_cos:.6f})")
    elif avg_cos > 0.98:
        print(f"\n✅ Acceptable batching (avg cos_sim: {avg_cos:.6f})")
    else:
        print(f"\n❌ Significant interference (avg cos_sim: {avg_cos:.6f})")
        return False
    
    print(f"\n{'=' * 70}")
    print("TEST 3: Batch of 8 images (larger batch)")
    print(f"{'=' * 70}")
    
    # Create 8 test images
    large_batch = test_images * 2  # 8 images
    
    start = time.time()
    batch_response = requests.post(
        f"{base_url}/img/batch",
        json={"contents": large_batch},
        timeout=60
    ).json()
    elapsed = time.time() - start
    
    batch_embs = batch_response["embeddings"]
    
    print(f"Batch size: {len(batch_embs)}")
    print(f"Server processing time: {batch_response['time_ms']:.2f}ms")
    print(f"Total request time: {elapsed*1000:.2f}ms")
    print(f"Throughput: {len(batch_embs) / (batch_response['time_ms']/1000):.2f} img/s")
    
    # Verify all embeddings are normalized
    norms = [np.linalg.norm(emb) for emb in batch_embs]
    all_normalized = all(abs(n - 1.0) < 0.01 for n in norms)
    
    if all_normalized:
        print("✅ All embeddings are L2-normalized")
    else:
        print(f"❌ Some embeddings not normalized: {[f'{n:.4f}' for n in norms[:3]]}...")
        return False
    
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print("✅ All batching tests passed!")
    print(f"   - 2-image batch: working")
    print(f"   - 4-image batch: working")
    print(f"   - 8-image batch: working")
    print(f"   - Embeddings normalized: working")
    
    return True


if __name__ == "__main__":
    base_url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080"
    success = test_batch_endpoint(base_url)
    sys.exit(0 if success else 1)

