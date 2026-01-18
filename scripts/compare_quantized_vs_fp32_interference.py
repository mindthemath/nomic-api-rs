#!/usr/bin/env python3
"""
Compare interference between quantized and fp32 (non-quantized) ONNX models.

Tests if quantization is the cause of cross-sample interference.
"""

import sys
from io import BytesIO
from pathlib import Path

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
    return np.dot(a, b)


def test_model(model_path: Path, model_name: str):
    """Test a specific model for interference."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'=' * 70}")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    session = ort.InferenceSession(str(model_path))

    # Load two different images
    image1 = load_image("https://picsum.photos/400/300")
    image2 = load_image("https://picsum.photos/300/400")

    tensor1 = preprocess_image(image1)
    tensor2 = preprocess_image(image2)

    # Single inference
    single_emb1 = embed_single(session, tensor1)
    single_emb2 = embed_single(session, tensor2)

    # Batch inference
    batch_embs = embed_batch(session, [tensor1, tensor2])

    # Calculate metrics
    max_diff_1 = np.abs(single_emb1 - batch_embs[0]).max()
    max_diff_2 = np.abs(single_emb2 - batch_embs[1]).max()
    mean_diff_1 = np.abs(single_emb1 - batch_embs[0]).mean()
    mean_diff_2 = np.abs(single_emb2 - batch_embs[1]).mean()
    cos_sim_1 = cosine_similarity(single_emb1, batch_embs[0])
    cos_sim_2 = cosine_similarity(single_emb2, batch_embs[1])
    l2_dist_1 = np.linalg.norm(single_emb1 - batch_embs[0])
    l2_dist_2 = np.linalg.norm(single_emb2 - batch_embs[1])

    return {
        "model_name": model_name,
        "max_diff_1": max_diff_1,
        "max_diff_2": max_diff_2,
        "mean_diff_1": mean_diff_1,
        "mean_diff_2": mean_diff_2,
        "cos_sim_1": cos_sim_1,
        "cos_sim_2": cos_sim_2,
        "l2_dist_1": l2_dist_1,
        "l2_dist_2": l2_dist_2,
    }


def main():
    print("=" * 70)
    print("Quantized vs FP32 Interference Comparison")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "img"

    quantized_path = models_dir / "model_quantized.onnx"
    fp32_path = models_dir / "model.onnx"

    results = {}

    # Test quantized model
    if quantized_path.exists():
        results["quantized"] = test_model(quantized_path, "Quantized (INT8)")
    else:
        print(f"\n⚠️  Quantized model not found: {quantized_path}")

    # Test fp32 model
    if fp32_path.exists():
        results["fp32"] = test_model(fp32_path, "FP32 (Full Precision)")
    else:
        print(f"\n⚠️  FP32 model not found: {fp32_path}")
        print(f"   Run: make model-img (downloads both)")

    # Compare results
    if len(results) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")

        q = results["quantized"]
        f = results["fp32"]

        print(f"\n{'Metric':<30} {'Quantized':<20} {'FP32':<20} {'Difference':<15}")
        print("-" * 85)

        print(
            f"{'Max diff (image 1)':<30} {q['max_diff_1']:<20.6f} {f['max_diff_1']:<20.6f} {abs(q['max_diff_1'] - f['max_diff_1']):<15.6f}"
        )
        print(
            f"{'Max diff (image 2)':<30} {q['max_diff_2']:<20.6f} {f['max_diff_2']:<20.6f} {abs(q['max_diff_2'] - f['max_diff_2']):<15.6f}"
        )
        print(
            f"{'Mean diff (image 1)':<30} {q['mean_diff_1']:<20.6f} {f['mean_diff_1']:<20.6f} {abs(q['mean_diff_1'] - f['mean_diff_1']):<15.6f}"
        )
        print(
            f"{'Mean diff (image 2)':<30} {q['mean_diff_2']:<20.6f} {f['mean_diff_2']:<20.6f} {abs(q['mean_diff_2'] - f['mean_diff_2']):<15.6f}"
        )
        print(
            f"{'Cosine sim (image 1)':<30} {q['cos_sim_1']:<20.6f} {f['cos_sim_1']:<20.6f} {abs(q['cos_sim_1'] - f['cos_sim_1']):<15.6f}"
        )
        print(
            f"{'Cosine sim (image 2)':<30} {q['cos_sim_2']:<20.6f} {f['cos_sim_2']:<20.6f} {abs(q['cos_sim_2'] - f['cos_sim_2']):<15.6f}"
        )
        print(
            f"{'L2 distance (image 1)':<30} {q['l2_dist_1']:<20.6f} {f['l2_dist_1']:<20.6f} {abs(q['l2_dist_1'] - f['l2_dist_1']):<15.6f}"
        )
        print(
            f"{'L2 distance (image 2)':<30} {q['l2_dist_2']:<20.6f} {f['l2_dist_2']:<20.6f} {abs(q['l2_dist_2'] - f['l2_dist_2']):<15.6f}"
        )

        print(f"\n{'=' * 70}")
        print("CONCLUSION")
        print(f"{'=' * 70}")

        if abs(q["max_diff_1"] - f["max_diff_1"]) < 0.001:
            print(
                "\n✅ Quantization is NOT the cause - interference is similar in both models"
            )
            print("   The interference is likely due to ONNX Runtime optimizations or")
            print("   numerical precision differences in batch processing.")
        elif q["max_diff_1"] > f["max_diff_1"] * 1.5:
            print("\n⚠️  Quantization appears to INCREASE interference")
            print("   Quantized model shows more interference than FP32.")
        else:
            print("\n❓ Results are mixed - further investigation needed")

    elif len(results) == 1:
        model_name = list(results.keys())[0]
        r = results[model_name]
        print(f"\n{'=' * 70}")
        print("RESULTS (Single Model)")
        print(f"{'=' * 70}")
        print(f"\n{model_name}:")
        print(f"  Max diff (image 1): {r['max_diff_1']:.6f}")
        print(f"  Max diff (image 2): {r['max_diff_2']:.6f}")
        print(
            f"  Cosine sim (image 1): {r['cos_sim_1']:.6f} ({r['cos_sim_1']*100:.2f}%)"
        )
        print(
            f"  Cosine sim (image 2): {r['cos_sim_2']:.6f} ({r['cos_sim_2']*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
