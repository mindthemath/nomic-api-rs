#!/usr/bin/env python3
"""
Benchmark vision model performance with different batch sizes.

Measures:
- Latency per image (single vs batched)
- Throughput (images/second)
- GPU vs CPU performance (if available)

Usage:
    source .venv/bin/activate
    python scripts/benchmark_vision_batching.py [--gpu] [--batch-sizes 1,4,8,16]
"""

import argparse
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


def create_test_image(seed: int = 42) -> Image.Image:
    """Create a synthetic test image."""
    np.random.seed(seed)
    r = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    g = np.tile(np.linspace(0, 255, 480, dtype=np.uint8).reshape(-1, 1), (1, 640))
    b = np.random.randint(0, 256, (480, 640), dtype=np.uint8)
    pixels = np.stack([r, g, b], axis=-1)
    return Image.fromarray(pixels, mode="RGB")


def benchmark_batch(
    session: ort.InferenceSession, tensors: list, warmup: int = 3, runs: int = 10
) -> dict:
    """Benchmark a batch of images."""
    batch_tensor = np.concatenate(tensors, axis=0)
    batch_size = len(tensors)

    # Warmup
    for _ in range(warmup):
        _ = session.run(None, {"pixel_values": batch_tensor})

    # Benchmark
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        result = session.run(None, {"pixel_values": batch_tensor})
        elapsed = (time.perf_counter() - start) * 1000  # ms
        times.append(elapsed)

    avg_time = np.mean(times)
    std_time = np.std(times)
    per_image_time = avg_time / batch_size
    throughput = (1000.0 / avg_time) * batch_size  # images/second

    return {
        "batch_size": batch_size,
        "total_time_ms": avg_time,
        "std_time_ms": std_time,
        "per_image_time_ms": per_image_time,
        "throughput_ips": throughput,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark vision model batching")
    parser.add_argument(
        "--gpu", action="store_true", help="Use GPU execution provider"
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,2,4,8,16",
        help="Comma-separated batch sizes to test",
    )
    parser.add_argument(
        "--warmup", type=int, default=3, help="Number of warmup runs"
    )
    parser.add_argument(
        "--runs", type=int, default=20, help="Number of benchmark runs"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("Vision Model Batching Benchmark")
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

    # Setup session
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if args.gpu else ["CPUExecutionProvider"]
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    print(f"\nLoading model: {model_path}")
    print(f"Execution providers: {providers}")
    try:
        session = ort.InferenceSession(
            str(model_path), sess_options=session_options, providers=providers
        )
        actual_providers = session.get_providers()
        print(f"Active providers: {actual_providers}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Create test images
    print(f"\nCreating test images...")
    test_image = create_test_image()
    tensors = [preprocess_image(test_image) for _ in range(32)]  # Pre-create enough

    # Parse batch sizes
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]

    print(f"\n{'=' * 70}")
    print("BENCHMARK RESULTS")
    print(f"{'=' * 70}")
    print(
        f"\n{'Batch':<8} {'Total (ms)':<12} {'Per Image (ms)':<16} {'Throughput (img/s)':<20} {'Speedup':<10}"
    )
    print("-" * 70)

    results = []
    baseline_time = None

    for batch_size in batch_sizes:
        if batch_size > len(tensors):
            print(f"⚠️  Skipping batch_size={batch_size} (not enough pre-created tensors)")
            continue

        batch_tensors = tensors[:batch_size]
        result = benchmark_batch(session, batch_tensors, args.warmup, args.runs)
        results.append(result)

        if baseline_time is None:
            baseline_time = result["per_image_time_ms"]
            speedup = 1.0
        else:
            speedup = baseline_time / result["per_image_time_ms"]

        print(
            f"{batch_size:<8} {result['total_time_ms']:>8.2f}±{result['std_time_ms']:>3.2f}  "
            f"{result['per_image_time_ms']:>10.2f}        {result['throughput_ips']:>12.2f}      "
            f"{speedup:>6.2f}x"
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")

    if len(results) > 1:
        best = max(results, key=lambda x: x["throughput_ips"])
        worst = min(results, key=lambda x: x["throughput_ips"])
        print(f"Best throughput: {best['throughput_ips']:.2f} img/s (batch_size={best['batch_size']})")
        print(f"Worst throughput: {worst['throughput_ips']:.2f} img/s (batch_size={worst['batch_size']})")

        if best["batch_size"] > 1:
            improvement = (best["throughput_ips"] / worst["throughput_ips"] - 1) * 100
            print(f"Batching improves throughput by {improvement:.1f}%")
        else:
            print("⚠️  Batching does not improve throughput (may be memory-bound)")

    print(f"\nRecommendations:")
    if len(results) > 1 and results[-1]["per_image_time_ms"] < results[0]["per_image_time_ms"]:
        optimal_batch = max(results, key=lambda x: x["throughput_ips"])
        print(f"  - Use batch_size={optimal_batch['batch_size']} for optimal throughput")
    else:
        print(f"  - Sequential processing (batch_size=1) is optimal")
        print(f"  - Consider horizontal scaling instead of batching")


if __name__ == "__main__":
    main()

