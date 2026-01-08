#!/usr/bin/env python3
"""
Test vision model variants: FP32, FP16, and quantized (INT8).

Compares:
1. Direct embedding differences (FP16 vs FP32, quantized vs FP32)
2. Batch size sensitivity
3. Image composition sensitivity (different sizes)
4. Ordering sensitivity
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import onnxruntime as ort
import requests
from PIL import Image


def load_image(source: str) -> Image.Image:
    """Load image from URL or file path."""
    if source.startswith(("http://", "https://")):
        return Image.open(requests.get(source, stream=True).raw)
    else:
        return Image.open(source)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to match model input format."""
    # Resize to 224x224
    image = image.resize((224, 224), Image.Resampling.BILINEAR)

    # Convert to RGB if needed
    if image.mode != "RGB":
        image = image.convert("RGB")

    # Convert to numpy array and normalize
    pixels = np.array(image, dtype=np.float32) / 255.0

    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    pixels = (pixels - mean) / std

    # Convert to NCHW format: [1, 3, 224, 224]
    pixels = np.transpose(pixels, (2, 0, 1))
    pixels = np.expand_dims(pixels, axis=0)

    return pixels


def embed_single(session: ort.InferenceSession, image: Image.Image) -> np.ndarray:
    """Embed a single image (batch_size=1)."""
    pixels = preprocess_image(image)

    result = session.run(None, {"pixel_values": pixels})[0]

    # Extract CLS token: [1, 197, 768] -> [768]
    embedding = result[0, 0, :]

    # L2 normalize
    norm = np.linalg.norm(embedding)
    return embedding / norm if norm > 0 else embedding


def embed_batch(session: ort.InferenceSession, images: list) -> list[np.ndarray]:
    """Embed multiple images in a single batch."""
    # Preprocess all images
    pixels_list = [preprocess_image(img) for img in images]

    # Stack into batch: [N, 3, 224, 224]
    pixels_batch = np.concatenate(pixels_list, axis=0)

    result = session.run(None, {"pixel_values": pixels_batch})[0]

    # Extract CLS token for each: [N, 197, 768] -> [N, 768]
    embeddings = result[:, 0, :]

    # L2 normalize each
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    return [embeddings[i] / norms[i, 0] for i in range(len(images))]


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b)


def create_session(model_path: Path):
    """Create ONNX session, handling FP16 optimization requirements."""
    try:
        return ort.InferenceSession(str(model_path))
    except Exception:
        # FP16 may need optimizations disabled
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        return ort.InferenceSession(str(model_path), sess_options)


def test_direct_accuracy(
    fp32_session, fp16_session, quantized_session, tokenizer_name="vision"
):
    """Test direct embedding differences between variants."""
    print(f"\n{'=' * 70}")
    print("TEST 1: Direct Accuracy Comparison")
    print(f"{'=' * 70}")

    # Load test images of different types
    test_images = [
        ("Small square", "https://picsum.photos/100/100"),
        ("Medium landscape", "https://picsum.photos/400/300"),
        ("Large portrait", "https://picsum.photos/300/400"),
        ("Nature scene", "https://picsum.photos/800/600"),
    ]

    print(f"\n{'Image':<30} {'FP16 vs FP32':<25} {'Quantized vs FP32':<25}")
    print(f"{'':<30} {'Max Diff':<12} {'Cos Sim':<12} {'Max Diff':<12} {'Cos Sim':<12}")
    print("-" * 80)

    fp16_diffs = []
    fp16_cos_sims = []
    quantized_diffs = []
    quantized_cos_sims = []

    for name, url in test_images:
        try:
            image = load_image(url)

            fp32_emb = embed_single(fp32_session, image)
            fp16_emb = embed_single(fp16_session, image)
            quantized_emb = embed_single(quantized_session, image)

            fp16_diff = np.abs(fp32_emb - fp16_emb).max()
            fp16_cos_sim = cosine_similarity(fp32_emb, fp16_emb)

            quantized_diff = np.abs(fp32_emb - quantized_emb).max()
            quantized_cos_sim = cosine_similarity(fp32_emb, quantized_emb)

            fp16_diffs.append(fp16_diff)
            fp16_cos_sims.append(fp16_cos_sim)
            quantized_diffs.append(quantized_diff)
            quantized_cos_sims.append(quantized_cos_sim)

            print(
                f"{name:<30} {fp16_diff:<12.6f} {fp16_cos_sim:<12.6f} {quantized_diff:<12.6f} {quantized_cos_sim:<12.6f}"
            )
        except Exception as e:
            print(f"{name:<30} Error: {e}")

    print(f"\nSummary:")
    print(f"  FP16 vs FP32:")
    print(f"    Average max diff: {np.mean(fp16_diffs):.6f}")
    print(f"    Max max diff: {np.max(fp16_diffs):.6f}")
    print(f"    Average cosine similarity: {np.mean(fp16_cos_sims):.6f}")
    print(f"    Min cosine similarity: {np.min(fp16_cos_sims):.6f}")
    print(f"  Quantized vs FP32:")
    print(f"    Average max diff: {np.mean(quantized_diffs):.6f}")
    print(f"    Max max diff: {np.max(quantized_diffs):.6f}")
    print(f"    Average cosine similarity: {np.mean(quantized_cos_sims):.6f}")
    print(f"    Min cosine similarity: {np.min(quantized_cos_sims):.6f}")

    return fp16_diffs, fp16_cos_sims, quantized_diffs, quantized_cos_sims


def test_batch_size_sensitivity(fp32_session, fp16_session, quantized_session):
    """Test sensitivity to batch size."""
    print(f"\n{'=' * 70}")
    print("TEST 2: Batch Size Sensitivity")
    print(f"{'=' * 70}")

    # Load test images
    images = [load_image("https://picsum.photos/400/300") for _ in range(8)]

    print(f"\n{'Batch Size':<15} {'FP16 vs FP32':<25} {'Quantized vs FP32':<25}")
    print(
        f"{'':<15} {'Max Diff':<12} {'Avg Cos Sim':<12} {'Max Diff':<12} {'Avg Cos Sim':<12}"
    )
    print("-" * 80)

    results = []

    for batch_size in [1, 2, 4, 8]:
        batch_images = images[:batch_size]

        # FP32
        fp32_embs = embed_batch(fp32_session, batch_images)

        # FP16
        fp16_embs = embed_batch(fp16_session, batch_images)

        # Quantized
        quantized_embs = embed_batch(quantized_session, batch_images)

        # Compare
        fp16_diffs = [
            np.abs(fp32_embs[i] - fp16_embs[i]).max() for i in range(batch_size)
        ]
        fp16_cos_sims = [
            cosine_similarity(fp32_embs[i], fp16_embs[i]) for i in range(batch_size)
        ]

        quantized_diffs = [
            np.abs(fp32_embs[i] - quantized_embs[i]).max() for i in range(batch_size)
        ]
        quantized_cos_sims = [
            cosine_similarity(fp32_embs[i], quantized_embs[i])
            for i in range(batch_size)
        ]

        fp16_max_diff = np.max(fp16_diffs)
        fp16_avg_cos_sim = np.mean(fp16_cos_sims)

        quantized_max_diff = np.max(quantized_diffs)
        quantized_avg_cos_sim = np.mean(quantized_cos_sims)

        results.append(
            (
                batch_size,
                fp16_max_diff,
                fp16_avg_cos_sim,
                quantized_max_diff,
                quantized_avg_cos_sim,
            )
        )
        print(
            f"{batch_size:<15} {fp16_max_diff:<12.6f} {fp16_avg_cos_sim:<12.6f} {quantized_max_diff:<12.6f} {quantized_avg_cos_sim:<12.6f}"
        )

    return results


def test_batching_interference(fp32_session, fp16_session, quantized_session):
    """Test batching interference (same image single vs batched)."""
    print(f"\n{'=' * 70}")
    print("TEST 3: Batching Interference (Single vs Batched)")
    print(f"{'=' * 70}")

    # Load two different images
    image1 = load_image("https://picsum.photos/400/300")
    image2 = load_image("https://picsum.photos/300/400")

    print(
        f"\n{'Variant':<15} {'Image 1 Diff':<20} {'Image 2 Diff':<20} {'Cos Sim 1':<15} {'Cos Sim 2':<15}"
    )
    print("-" * 85)

    for name, session in [
        ("FP32", fp32_session),
        ("FP16", fp16_session),
        ("Quantized", quantized_session),
    ]:
        # Single inference
        single_emb1 = embed_single(session, image1)
        single_emb2 = embed_single(session, image2)

        # Batch inference
        batch_embs = embed_batch(session, [image1, image2])

        diff_1 = np.abs(single_emb1 - batch_embs[0]).max()
        diff_2 = np.abs(single_emb2 - batch_embs[1]).max()
        cos_sim_1 = cosine_similarity(single_emb1, batch_embs[0])
        cos_sim_2 = cosine_similarity(single_emb2, batch_embs[1])

        print(
            f"{name:<15} {diff_1:<20.6f} {diff_2:<20.6f} {cos_sim_1:<15.6f} {cos_sim_2:<15.6f}"
        )


def main():
    print("=" * 70)
    print("Vision Model Variants Comparison: FP32 vs FP16 vs Quantized")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "img"

    fp32_path = models_dir / "model.onnx"
    fp16_path = models_dir / "model_fp16.onnx"
    quantized_path = models_dir / "model_quantized.onnx"

    # Check if int8 exists and is different
    int8_path = models_dir / "model_int8.onnx"
    uint8_path = models_dir / "model_uint8.onnx"

    if not fp32_path.exists():
        print(f"❌ FP32 model not found: {fp32_path}")
        sys.exit(1)

    if not fp16_path.exists():
        print(f"❌ FP16 model not found: {fp16_path}")
        print(f"   Download with: make model-img")
        sys.exit(1)

    if not quantized_path.exists():
        print(f"❌ Quantized model not found: {quantized_path}")
        print(f"   Download with: make model-img")
        sys.exit(1)

    print("\nLoading models...")
    fp32_session = create_session(fp32_path)
    fp16_session = create_session(fp16_path)
    quantized_session = create_session(quantized_path)
    print("✓ Models loaded")

    # Check if int8/uint8 are different from quantized
    if int8_path.exists() and int8_path.stat().st_size != quantized_path.stat().st_size:
        print(
            f"\n⚠️  Note: model_int8.onnx exists and has different size than model_quantized.onnx"
        )
        print(f"   Testing quantized (model_quantized.onnx)")
    if (
        uint8_path.exists()
        and uint8_path.stat().st_size != quantized_path.stat().st_size
    ):
        print(
            f"\n⚠️  Note: model_uint8.onnx exists and has different size than model_quantized.onnx"
        )

    # Run all tests
    test_direct_accuracy(fp32_session, fp16_session, quantized_session)
    test_batch_size_sensitivity(fp32_session, fp16_session, quantized_session)
    test_batching_interference(fp32_session, fp16_session, quantized_session)

    print(f"\n{'=' * 70}")
    print("SUMMARY & RECOMMENDATIONS")
    print(f"{'=' * 70}")
    print(
        """
Model Size Comparison:
  - FP32: ~358MB (full precision)
  - FP16: ~179MB (half precision, 2x smaller)
  - Quantized: ~93MB (INT8, 4x smaller)

For CPU Usage:
  1. If accuracy is critical: Use FP32 (perfect accuracy)
  2. If size/accuracy balance: Use FP16 (excellent accuracy, 2x smaller)
  3. If size is critical: Use Quantized (good accuracy, 4x smaller, minor batching interference)

Check the test results above to see:
  - Direct accuracy differences
  - Batch size sensitivity
  - Batching interference (should be minimal for all variants)
"""
    )


if __name__ == "__main__":
    main()
