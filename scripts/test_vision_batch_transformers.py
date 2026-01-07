#!/usr/bin/env python3
"""
Test script to check if transformers implementation shows cross-sample interference.

This tests the PyTorch/transformers implementation directly (not ONNX) to see if
the interference is a model characteristic or ONNX-specific.

Usage:
    source .venv/bin/activate
    python scripts/test_vision_batch_transformers.py [image_url]
"""

import sys
import time
from pathlib import Path
from io import BytesIO

import torch
import torch.nn.functional as F
from PIL import Image

try:
    from transformers import AutoImageProcessor, AutoModel
except ImportError:
    print("Missing: pip install transformers torch")
    sys.exit(1)


def load_image(source: str) -> Image.Image:
    """Load image from path or URL."""
    if source.startswith(("http://", "https://")):
        import requests

        response = requests.get(source, timeout=30)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    elif Path(source).exists():
        return Image.open(source)
    else:
        raise ValueError(f"Cannot load image: {source}")


def embed_single(processor, model, image: Image.Image, device: str) -> torch.Tensor:
    """Embed a single image (batch_size=1)."""
    inputs = processor(image, return_tensors="pt")
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_emb = model(**inputs).last_hidden_state
        # Extract CLS token: [1, 197, 768] → [1, 768]
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
    return img_embeddings[0]  # Return [768] tensor


def embed_batch(processor, model, images: list, device: str) -> list[torch.Tensor]:
    """Embed multiple images in a single batch."""
    inputs = processor(images, return_tensors="pt")
    # Move inputs to same device as model
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        img_emb = model(**inputs).last_hidden_state
        # Extract CLS token for all images: [N, 197, 768] → [N, 768]
        img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
    # Return list of [768] tensors
    return [img_embeddings[i] for i in range(len(images))]


def main():
    print("=" * 70)
    print("Vision Model Batching Test (Transformers/PyTorch)")
    print("Model: nomic-embed-vision-v1.5")
    print("=" * 70)

    # Load model and processor
    print("\nLoading model and processor...")
    print("(This may take a minute on first run - downloads from HuggingFace)")
    
    try:
        processor = AutoImageProcessor.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5"
        )
        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
        )
        model.eval()  # Set to evaluation mode
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Using device: {device}")

    # Load test images
    if len(sys.argv) > 1:
        image_source = sys.argv[1]
    else:
        image_source = "https://picsum.photos/400/300"

    print(f"\nLoading test images: {image_source}")
    try:
        image1 = load_image(image_source)
        # Create a second different image
        if image_source.startswith(("http://", "https://")):
            image2 = load_image("https://picsum.photos/300/400")
        else:
            image2 = image1.copy()
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)

    print(f"Image 1 size: {image1.size}")
    print(f"Image 2 size: {image2.size}")

    # Test 1: Same image, single vs batched
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 1: Same image, single vs batched")
    print(f"{'=' * 70}")

    ref_emb = embed_single(processor, model, image1, device)
    print(f"Reference (image1 alone): {ref_emb[:3].cpu().numpy().round(4)}")

    # Batch: same image × 2
    batch_embs = embed_batch(processor, model, [image1, image1], device)
    
    diff_0 = torch.abs(ref_emb - batch_embs[0]).max().item()
    diff_1 = torch.abs(ref_emb - batch_embs[1]).max().item()
    diff_batch = torch.abs(batch_embs[0] - batch_embs[1]).max().item()

    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)
    status_0 = "✓ identical" if diff_0 < 0.0001 else "✗ DIFFERENT"
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_batch = "✓ identical" if diff_batch < 0.0001 else "✗ DIFFERENT"
    print(f"{'Single vs batch[0]':<40} {diff_0:.6f} {status_0}")
    print(f"{'Single vs batch[1]':<40} {diff_1:.6f} {status_1}")
    print(f"{'batch[0] vs batch[1]':<40} {diff_batch:.6f} {status_batch}")

    # Test 2: Different images in batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: Different images in batch")
    print(f"{'=' * 70}")

    # Single inference for both images
    single_emb1 = embed_single(processor, model, image1, device)
    single_emb2 = embed_single(processor, model, image2, device)

    # Batch inference
    batch_embs = embed_batch(processor, model, [image1, image2], device)

    diff_1 = torch.abs(single_emb1 - batch_embs[0]).max().item()
    diff_2 = torch.abs(single_emb2 - batch_embs[1]).max().item()

    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_2 = "✓ identical" if diff_2 < 0.0001 else "✗ DIFFERENT"
    print(f"{'image1 single vs batch[0]':<40} {diff_1:.6f} {status_1}")
    print(f"{'image2 single vs batch[1]':<40} {diff_2:.6f} {status_2}")

    # Test 3: Larger batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: Larger batch (4 images)")
    print(f"{'=' * 70}")

    single_embs = [embed_single(processor, model, image1, device) for _ in range(4)]
    batch_embs = embed_batch(processor, model, [image1, image1, image1, image1], device)

    max_diff = max(
        torch.abs(single_embs[i] - batch_embs[i]).max().item() for i in range(4)
    )
    print(f"Max difference across all 4 images: {max_diff:.6f}")
    status = "✓ identical" if max_diff < 0.0001 else "✗ DIFFERENT"
    print(f"Result: {status}")

    # Test 4: Compare with ONNX results
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH ONNX RESULTS")
    print(f"{'=' * 70}")
    print(f"\nONNX results showed:")
    print(f"  - Same image × 2: 0.000000 (identical)")
    print(f"  - Different images: ~0.016-0.018 (interference)")
    print(f"  - Same image × 4: 0.000000 (identical)")
    print(f"\nTransformers results:")
    print(f"  - Same image × 2: {diff_0:.6f} (identical: {diff_0 < 0.0001})")
    print(f"  - Different images: {diff_1:.6f}, {diff_2:.6f}")
    print(f"  - Same image × 4: {max_diff:.6f} (identical: {max_diff < 0.0001})")

    # Conclusion
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")

    all_identical_same = diff_0 < 0.0001 and diff_1 < 0.0001 and diff_batch < 0.0001 and max_diff < 0.0001
    interference_different = diff_1 > 0.001 or diff_2 > 0.001

    if all_identical_same and not interference_different:
        print(
            """
✅ Transformers implementation shows NO interference - batching is safe!

This suggests the interference in ONNX is specific to the ONNX export/runtime,
not a fundamental model characteristic. The transformers implementation can
safely batch different images.
"""
        )
    elif all_identical_same and interference_different:
        print(
            """
⚠️  Transformers implementation shows SAME behavior as ONNX:
    - Same images batch safely (no interference)
    - Different images show interference (~{:.4f})

This confirms the interference is a model characteristic, not ONNX-specific.
Both implementations show the same cross-sample interference pattern.
""".format(max(diff_1, diff_2))
        )
    else:
        print(
            """
❓ Unexpected results - further investigation needed.
"""
        )


if __name__ == "__main__":
    main()

