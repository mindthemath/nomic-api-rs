#!/usr/bin/env python3
"""
Analyze the interference metric - show what 0.016-0.022 difference means.

The metric is: max absolute difference per dimension between embeddings.
We'll also show cosine similarity to give context.
"""

import sys
from pathlib import Path
from io import BytesIO

import numpy as np
from PIL import Image

try:
    import onnxruntime as ort
except ImportError:
    print("Missing: pip install onnxruntime")
    sys.exit(1)

# Constants
IMAGE_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to model input format: [1, 3, 224, 224]."""
    if image.mode != "RGB":
        image = image.convert("RGB")
    w, h = image.size
    if w < h:
        new_w, new_h = IMAGE_SIZE, int(h * IMAGE_SIZE / w)
    else:
        new_w, new_h = int(w * IMAGE_SIZE / h), IMAGE_SIZE
    image = image.resize((new_w, new_h), Image.BICUBIC)
    w, h = image.size
    left = (w - IMAGE_SIZE) // 2
    top = (h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    pixels = np.array(image, dtype=np.float32) / 255.0
    pixels = (pixels - MEAN) / STD
    pixels = pixels.transpose(2, 0, 1)
    pixels = np.expand_dims(pixels, axis=0)
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
    if result.shape == (1, 768):
        embedding = result[0]
    elif len(result.shape) == 3 and result.shape[2] == 768:
        embedding = result[0, 0, :]
    else:
        raise ValueError(f"Unexpected output shape: {result.shape}")
    norm = np.linalg.norm(embedding)
    if norm > 1e-9:
        embedding = embedding / norm
    return embedding


def embed_batch(session: ort.InferenceSession, tensors: list) -> list[np.ndarray]:
    """Embed multiple images in a single batch."""
    batch_tensor = np.concatenate(tensors, axis=0)
    result = session.run(None, {"pixel_values": batch_tensor})[0]
    embeddings = []
    for i in range(len(tensors)):
        if result.shape == (len(tensors), 768):
            emb = result[i]
        elif len(result.shape) == 3 and result.shape[2] == 768:
            emb = result[i, 0, :]
        else:
            raise ValueError(f"Unexpected output shape: {result.shape}")
        norm = np.linalg.norm(emb)
        if norm > 1e-9:
            emb = emb / norm
        embeddings.append(emb)
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two normalized vectors."""
    return np.dot(a, b)  # Already normalized, so dot product = cosine


def analyze_difference(emb1: np.ndarray, emb2: np.ndarray, label: str):
    """Analyze the difference between two embeddings."""
    # Max absolute difference per dimension
    max_diff = np.abs(emb1 - emb2).max()
    
    # Mean absolute difference
    mean_diff = np.abs(emb1 - emb2).mean()
    
    # L2 distance (Euclidean distance)
    l2_dist = np.linalg.norm(emb1 - emb2)
    
    # Cosine similarity (embeddings are L2-normalized)
    cos_sim = cosine_similarity(emb1, emb2)
    
    # Number of dimensions that differ significantly (>0.001)
    num_different = np.sum(np.abs(emb1 - emb2) > 0.001)
    
    print(f"\n{label}:")
    print(f"  Max absolute difference (per dimension): {max_diff:.6f}")
    print(f"  Mean absolute difference: {mean_diff:.6f}")
    print(f"  L2 distance: {l2_dist:.6f}")
    print(f"  Cosine similarity: {cos_sim:.6f} ({cos_sim*100:.4f}%)")
    print(f"  Dimensions differing >0.001: {num_different}/768 ({num_different/768*100:.1f}%)")
    
    # Show distribution of differences
    diffs = np.abs(emb1 - emb2)
    print(f"  Difference distribution:")
    print(f"    Min: {diffs.min():.6f}")
    print(f"    P50 (median): {np.median(diffs):.6f}")
    print(f"    P95: {np.percentile(diffs, 95):.6f}")
    print(f"    P99: {np.percentile(diffs, 99):.6f}")
    print(f"    Max: {diffs.max():.6f}")


def main():
    print("=" * 70)
    print("Interference Metric Analysis")
    print("=" * 70)
    
    # Load model
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_path = project_root / "models" / "img" / "model_quantized.onnx"
    if not model_path.exists():
        model_path = project_root / "models" / "img" / "model.onnx"
    if not model_path.exists():
        print(f"❌ Model not found. Run: make model-img")
        sys.exit(1)
    
    print(f"\nLoading model: {model_path}")
    session = ort.InferenceSession(str(model_path))
    
    # Load two different images
    print("\nLoading test images...")
    image1 = load_image("https://picsum.photos/400/300")
    image2 = load_image("https://picsum.photos/300/400")
    
    tensor1 = preprocess_image(image1)
    tensor2 = preprocess_image(image2)
    
    print("\n" + "=" * 70)
    print("EXPERIMENT: Different images - single vs batched")
    print("=" * 70)
    
    # Single inference
    print("\nProcessing images individually (batch_size=1)...")
    single_emb1 = embed_single(session, tensor1)
    single_emb2 = embed_single(session, tensor2)
    
    # Batch inference
    print("Processing images in batch (batch_size=2)...")
    batch_embs = embed_batch(session, [tensor1, tensor2])
    
    # Analyze differences
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)
    
    analyze_difference(single_emb1, batch_embs[0], "Image 1: Single vs Batched")
    analyze_difference(single_emb2, batch_embs[1], "Image 2: Single vs Batched")
    
    # Also compare the two images to each other (baseline)
    print("\n" + "=" * 70)
    print("BASELINE: Image 1 vs Image 2 (different images, both single)")
    print("=" * 70)
    analyze_difference(single_emb1, single_emb2, "Different images (expected to be different)")
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    max_diff_1 = np.abs(single_emb1 - batch_embs[0]).max()
    max_diff_2 = np.abs(single_emb2 - batch_embs[1]).max()
    cos_sim_1 = cosine_similarity(single_emb1, batch_embs[0])
    cos_sim_2 = cosine_similarity(single_emb2, batch_embs[1])
    
    print(f"\nInterference metric (max absolute difference):")
    print(f"  Image 1: {max_diff_1:.6f}")
    print(f"  Image 2: {max_diff_2:.6f}")
    
    print(f"\nCosine similarity (should be 1.0 if identical):")
    print(f"  Image 1: {cos_sim_1:.6f} (differs by {1-cos_sim_1:.6f})")
    print(f"  Image 2: {cos_sim_2:.6f} (differs by {1-cos_sim_2:.6f})")
    
    print(f"\nInterpretation:")
    print(f"  - Max diff of ~0.02 means at least one dimension differs by 0.02")
    print(f"  - Cosine similarity of ~{cos_sim_1:.6f} means vectors are {cos_sim_1*100:.4f}% similar")
    print(f"  - For comparison, two completely different images have cosine ~0.3-0.7")
    print(f"  - The interference is small but measurable")


if __name__ == "__main__":
    main()

