#!/usr/bin/env python3
"""
Test script to check if vision model batching causes cross-sample interference.

Unlike the text model, vision models should be able to batch safely since:
1. All images are preprocessed to the same size (224x224)
2. CLS token extraction is per-sample (index [batch_idx, 0, :])
3. Vision transformers process each image independently

Usage:
    source .venv/bin/activate  # if using venv with onnxruntime
    python scripts/test_vision_batch_interference.py [image_path_or_url]
"""

import sys
import time
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Missing: pip install onnxruntime")
    sys.exit(1)

# Constants from preprocessor_config.json
IMAGE_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to model input format: [1, 3, 224, 224]."""
    # Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Resize: shortest edge to 224, maintain aspect ratio
    w, h = image.size
    if w < h:
        new_w, new_h = IMAGE_SIZE, int(h * IMAGE_SIZE / w)
    else:
        new_w, new_h = int(w * IMAGE_SIZE / h), IMAGE_SIZE
    image = image.resize((new_w, new_h), Image.BICUBIC)

    # Center crop to 224x224
    w, h = image.size
    left = (w - IMAGE_SIZE) // 2
    top = (h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))

    # Convert to numpy array and normalize
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD

    # HWC → CHW → NCHW
    pixels = pixels.transpose(2, 0, 1)  # (3, 224, 224)
    pixels = np.expand_dims(pixels, axis=0)  # (1, 3, 224, 224)

    return pixels


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


def embed_single(session: ort.InferenceSession, tensor: np.ndarray) -> np.ndarray:
    """Embed a single image (batch_size=1)."""
    result = session.run(None, {"pixel_values": tensor})[0]
    # Extract CLS token: [1, 197, 768] or [1, 768] → [768]
    if result.shape == (1, 768):
        embedding = result[0]
    elif len(result.shape) == 3 and result.shape[2] == 768:
        embedding = result[0, 0, :]  # CLS token is first token
    else:
        raise ValueError(f"Unexpected output shape: {result.shape}")

    # L2 normalize
    norm = np.linalg.norm(embedding)
    if norm > 1e-9:
        embedding = embedding / norm
    return embedding


def embed_batch(session: ort.InferenceSession, tensors: list) -> list[np.ndarray]:
    """Embed multiple images in a single batch."""
    # Stack tensors: [N, 3, 224, 224]
    batch_tensor = np.concatenate(tensors, axis=0)

    result = session.run(None, {"pixel_values": batch_tensor})[0]
    # Output shape: [N, 197, 768] or [N, 768]

    embeddings = []
    for i in range(len(tensors)):
        if result.shape == (len(tensors), 768):
            emb = result[i]
        elif len(result.shape) == 3 and result.shape[2] == 768:
            emb = result[i, 0, :]  # CLS token per sample
        else:
            raise ValueError(f"Unexpected output shape: {result.shape}")

        # L2 normalize
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        embeddings.append(emb)

    return embeddings


def main():
    print("=" * 70)
    print("Vision Model Batching Test for nomic-embed-vision-v1.5")
    print("=" * 70)

    # Load model (try relative to script, then relative to project root)
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    model_path = project_root / "models" / "img" / "model_quantized.onnx"
    if not model_path.exists():
        model_path = project_root / "models" / "img" / "model.onnx"
    if not model_path.exists():
        print(f"❌ Model not found at {model_path}")
        print(f"   Run: make model-img")
        sys.exit(1)

    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Get execution provider info
    providers = session.get_providers()
    print(f"Execution providers: {providers}")

    # Load test images
    if len(sys.argv) > 1:
        image_source = sys.argv[1]
    else:
        # Use a test image URL
        image_source = "https://picsum.photos/400/300"

    print(f"\nLoading test image: {image_source}")
    try:
        image1 = load_image(image_source)
        # Create a second different image (or use same one)
        if image_source.startswith(("http://", "https://")):
            # For URL, use a different one
            image2 = load_image("https://picsum.photos/300/400")
        else:
            # For file, use same image (will test identical case)
            image2 = image1.copy()
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        sys.exit(1)

    # Preprocess
    tensor1 = preprocess_image(image1)
    tensor2 = preprocess_image(image2)

    print(f"\nPreprocessed tensors:")
    print(f"  Image 1: {tensor1.shape}")
    print(f"  Image 2: {tensor2.shape}")

    # Reference: single inference for image 1
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 1: Same image, single vs batched")
    print(f"{'=' * 70}")
    ref_emb = embed_single(session, tensor1)
    print(f"Reference (image1 alone): {ref_emb[:3].round(4)}")

    # Test: same image × 2 in batch
    batch_embs = embed_batch(session, [tensor1, tensor1])
    diff_0 = np.abs(ref_emb - batch_embs[0]).max()
    diff_1 = np.abs(ref_emb - batch_embs[1]).max()
    diff_batch = np.abs(batch_embs[0] - batch_embs[1]).max()

    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)
    status_0 = "✓ identical" if diff_0 < 0.0001 else "✗ DIFFERENT"
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_batch = "✓ identical" if diff_batch < 0.0001 else "✗ DIFFERENT"
    print(f"{'Single vs batch[0]':<40} {diff_0:.6f} {status_0}")
    print(f"{'Single vs batch[1]':<40} {diff_1:.6f} {status_1}")
    print(f"{'batch[0] vs batch[1]':<40} {diff_batch:.6f} {status_batch}")

    # Test: different images in batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: Different images in batch")
    print(f"{'=' * 70}")

    # Single inference for both images
    single_emb1 = embed_single(session, tensor1)
    single_emb2 = embed_single(session, tensor2)

    # Batch inference
    batch_embs = embed_batch(session, [tensor1, tensor2])

    diff_1 = np.abs(single_emb1 - batch_embs[0]).max()
    diff_2 = np.abs(single_emb2 - batch_embs[1]).max()

    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_2 = "✓ identical" if diff_2 < 0.0001 else "✗ DIFFERENT"
    print(f"{'image1 single vs batch[0]':<40} {diff_1:.6f} {status_1}")
    print(f"{'image2 single vs batch[1]':<40} {diff_2:.6f} {status_2}")

    # Test: larger batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: Larger batch (4 images)")
    print(f"{'=' * 70}")

    single_embs = [embed_single(session, tensor1) for _ in range(4)]
    batch_embs = embed_batch(session, [tensor1, tensor1, tensor1, tensor1])

    max_diff = max(
        np.abs(single_embs[i] - batch_embs[i]).max() for i in range(4)
    )
    print(f"Max difference across all 4 images: {max_diff:.6f}")
    status = "✓ identical" if max_diff < 0.0001 else "✗ DIFFERENT"
    print(f"Result: {status}")

    # Conclusion
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")

    all_identical = (
        diff_0 < 0.0001
        and diff_1 < 0.0001
        and diff_batch < 0.0001
        and max_diff < 0.0001
    )

    if all_identical:
        print(
            """
✅ Vision model batching is SAFE - no cross-sample interference detected.

The model can process multiple images in a single batch without affecting
individual embeddings. This means batching can be used to improve GPU
throughput without sacrificing correctness.

Recommendation: Implement batched inference for /img/batch endpoint.
"""
        )
    else:
        print(
            """
❌ Vision model batching shows INTERFERENCE - embeddings differ between
single and batched inference.

This suggests the model has cross-sample dependencies similar to the text
model. Sequential processing (batch_size=1) is required for deterministic
results.

Further investigation needed to understand the root cause.
"""
        )


if __name__ == "__main__":
    main()

