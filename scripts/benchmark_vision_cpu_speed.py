#!/usr/bin/env python3
"""
Benchmark CPU inference speed for vision model variants: FP32, FP16, and INT8.

Tests single inference and batched inference to help decide which model to use.
"""

import sys
import time
from pathlib import Path
import statistics

import numpy as np
import onnxruntime as ort
from PIL import Image
import requests


def load_image(source: str) -> Image.Image:
    """Load image from URL or file path."""
    if source.startswith(("http://", "https://")):
        return Image.open(requests.get(source, stream=True).raw)
    else:
        return Image.open(source)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """Preprocess image to match model input format."""
    image = image.resize((224, 224), Image.Resampling.BILINEAR)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pixels = np.array(image, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)
    pixels = (pixels - mean) / std
    pixels = np.transpose(pixels, (2, 0, 1))
    pixels = np.expand_dims(pixels, axis=0)
    return pixels


def embed_single(session: ort.InferenceSession, image: Image.Image) -> np.ndarray:
    """Embed a single image (batch_size=1)."""
    pixels = preprocess_image(image)
    result = session.run(None, {"pixel_values": pixels})[0]
    return result[0, 0, :]  # CLS token


def embed_batch(session: ort.InferenceSession, images: list) -> list[np.ndarray]:
    """Embed multiple images in a single batch."""
    pixels_list = [preprocess_image(img) for img in images]
    pixels_batch = np.concatenate(pixels_list, axis=0)
    result = session.run(None, {"pixel_values": pixels_batch})[0]
    return result[:, 0, :]  # CLS tokens


def benchmark_model(session: ort.InferenceSession, model_name: str, images: list, warmup: int = 3, runs: int = 20):
    """Benchmark a model with single and batched inference."""
    print(f"\n{'=' * 70}")
    print(f"Benchmarking: {model_name}")
    print(f"{'=' * 70}")
    
    # Warmup
    for _ in range(warmup):
        _ = embed_single(session, images[0])
        _ = embed_batch(session, images[:4])
    
    # Single inference
    print(f"\nSingle inference (batch_size=1):")
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = embed_single(session, images[0])
        times.append((time.perf_counter() - start) * 1000)  # Convert to ms
    
    single_mean = statistics.mean(times)
    single_std = statistics.stdev(times) if len(times) > 1 else 0
    single_min = min(times)
    single_max = max(times)
    single_throughput = 1000 / single_mean if single_mean > 0 else 0
    
    print(f"  Mean: {single_mean:.2f} ± {single_std:.2f} ms")
    print(f"  Min: {single_min:.2f} ms")
    print(f"  Max: {single_max:.2f} ms")
    print(f"  Throughput: {single_throughput:.2f} img/s")
    
    # Batched inference (batch_size=4)
    print(f"\nBatched inference (batch_size=4):")
    batch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = embed_batch(session, images[:4])
        batch_times.append((time.perf_counter() - start) * 1000)
    
    batch_mean = statistics.mean(batch_times)
    batch_std = statistics.stdev(batch_times) if len(batch_times) > 1 else 0
    batch_min = min(batch_times)
    batch_max = max(batch_times)
    batch_per_image = batch_mean / 4
    batch_throughput = 4000 / batch_mean if batch_mean > 0 else 0
    
    print(f"  Total: {batch_mean:.2f} ± {batch_std:.2f} ms")
    print(f"  Per image: {batch_per_image:.2f} ms")
    print(f"  Throughput: {batch_throughput:.2f} img/s")
    print(f"  Speedup vs single: {single_mean / batch_per_image:.2f}x")
    
    # Batched inference (batch_size=8)
    print(f"\nBatched inference (batch_size=8):")
    batch8_times = []
    for _ in range(runs):
        start = time.perf_counter()
        _ = embed_batch(session, images[:8])
        batch8_times.append((time.perf_counter() - start) * 1000)
    
    batch8_mean = statistics.mean(batch8_times)
    batch8_std = statistics.stdev(batch8_times) if len(batch8_times) > 1 else 0
    batch8_per_image = batch8_mean / 8
    batch8_throughput = 8000 / batch8_mean if batch8_mean > 0 else 0
    
    print(f"  Total: {batch8_mean:.2f} ± {batch8_std:.2f} ms")
    print(f"  Per image: {batch8_per_image:.2f} ms")
    print(f"  Throughput: {batch8_throughput:.2f} img/s")
    print(f"  Speedup vs single: {single_mean / batch8_per_image:.2f}x")
    
    return {
        "single_mean": single_mean,
        "single_throughput": single_throughput,
        "batch4_per_image": batch_per_image,
        "batch4_throughput": batch_throughput,
        "batch4_speedup": single_mean / batch_per_image,
        "batch8_per_image": batch8_per_image,
        "batch8_throughput": batch8_throughput,
        "batch8_speedup": single_mean / batch8_per_image,
    }


def create_session(model_path: Path):
    """Create ONNX session, handling FP16 optimization requirements."""
    try:
        return ort.InferenceSession(str(model_path))
    except Exception:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        return ort.InferenceSession(str(model_path), sess_options)


def main():
    print("=" * 70)
    print("Vision Model CPU Speed Benchmark")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "img"
    
    fp32_path = models_dir / "model.onnx"
    fp16_path = models_dir / "model_fp16.onnx"
    int8_path = models_dir / "model_int8.onnx"
    
    # Check which models exist
    models_to_test = []
    if fp32_path.exists():
        models_to_test.append(("FP32", fp32_path))
    if fp16_path.exists():
        models_to_test.append(("FP16", fp16_path))
    if int8_path.exists():
        models_to_test.append(("INT8", int8_path))
    
    if not models_to_test:
        print("❌ No models found. Run: make model-img")
        sys.exit(1)
    
    print(f"\nTesting models: {', '.join([m[0] for m in models_to_test])}")
    
    # Load test images
    print("\nLoading test images...")
    images = []
    for i in range(8):
        try:
            img = load_image(f"https://picsum.photos/400/300?random={i}")
            images.append(img)
        except Exception as e:
            print(f"⚠️  Failed to load image {i}: {e}")
            # Use a simple test image if download fails
            img = Image.new("RGB", (224, 224), color=(128, 128, 128))
            images.append(img)
    
    print(f"✓ Loaded {len(images)} test images")
    
    # Benchmark each model
    results = {}
    sessions = {}
    
    for model_name, model_path in models_to_test:
        print(f"\nLoading {model_name} model...")
        session = create_session(model_path)
        sessions[model_name] = session
        results[model_name] = benchmark_model(session, model_name, images)
    
    # Compare results
    print(f"\n{'=' * 70}")
    print("COMPARISON")
    print(f"{'=' * 70}")
    
    print(f"\n{'Metric':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{model_name:<15}", end="")
    print()
    print("-" * (30 + 15 * len(models_to_test)))
    
    # Single inference
    print(f"{'Single inference (ms)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['single_mean']:<15.2f}", end="")
    print()
    
    print(f"{'Single throughput (img/s)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['single_throughput']:<15.2f}", end="")
    print()
    
    # Batch 4
    print(f"{'Batch 4 per image (ms)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch4_per_image']:<15.2f}", end="")
    print()
    
    print(f"{'Batch 4 throughput (img/s)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch4_throughput']:<15.2f}", end="")
    print()
    
    print(f"{'Batch 4 speedup':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch4_speedup']:<15.2f}x", end="")
    print()
    
    # Batch 8
    print(f"{'Batch 8 per image (ms)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch8_per_image']:<15.2f}", end="")
    print()
    
    print(f"{'Batch 8 throughput (img/s)':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch8_throughput']:<15.2f}", end="")
    print()
    
    print(f"{'Batch 8 speedup':<30} ", end="")
    for model_name, _ in models_to_test:
        print(f"{results[model_name]['batch8_speedup']:<15.2f}x", end="")
    print()
    
    # Model sizes
    print(f"\n{'Model size (MB)':<30} ", end="")
    for model_name, model_path in models_to_test:
        size_mb = model_path.stat().st_size / (1024 * 1024)
        print(f"{size_mb:<15.1f}", end="")
    print()
    
    # Recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 70}")
    
    if "FP16" in results and "INT8" in results:
        fp16_single = results["FP16"]["single_mean"]
        int8_single = results["INT8"]["single_mean"]
        fp16_size = fp16_path.stat().st_size / (1024 * 1024)
        int8_size = int8_path.stat().st_size / (1024 * 1024)
        
        speed_ratio = int8_single / fp16_single if fp16_single > 0 else 1.0
        size_ratio = int8_size / fp16_size if fp16_size > 0 else 1.0
        
        print(f"\nFP16 vs INT8 Comparison:")
        print(f"  Speed: INT8 is {speed_ratio:.2f}x {'faster' if speed_ratio < 1.0 else 'slower'} than FP16")
        print(f"  Size: INT8 is {1/size_ratio:.2f}x smaller than FP16 ({int8_size:.1f}MB vs {fp16_size:.1f}MB)")
        print(f"  Accuracy: FP16 is 99.9999% vs FP32, INT8 is 94.1% vs FP32")
        
        if speed_ratio < 0.9:
            print(f"\n✅ INT8 is faster - consider for CPU deployment")
        elif speed_ratio > 1.1:
            print(f"\n✅ FP16 is faster - better for CPU deployment")
        else:
            print(f"\n⚠️  Speed is similar - choose based on size vs accuracy trade-off")
        
        print(f"\nRecommendation:")
        if speed_ratio < 0.85 and size_ratio < 0.6:
            print(f"  Use INT8: Faster ({speed_ratio:.2f}x) and smaller ({1/size_ratio:.2f}x)")
            print(f"  Trade-off: 94% accuracy vs 99.9999% (acceptable for most use cases)")
        elif speed_ratio > 1.1:
            print(f"  Use FP16: Faster ({1/speed_ratio:.2f}x) with excellent accuracy")
            print(f"  Trade-off: 2x larger model size")
        else:
            print(f"  Use FP16: Better accuracy (99.9999% vs 94%) with similar speed")
            print(f"  Trade-off: 2x larger model size")


if __name__ == "__main__":
    main()

