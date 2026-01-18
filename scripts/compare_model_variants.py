#!/usr/bin/env python3
"""
Compare embeddings from different ONNX model variants.

Tests model_quantized.onnx (baseline) against:
- model_q4f16.onnx (4-bit quantized)
- model_fp16.onnx (FP16)
- model.onnx (FP32, full precision)

Measures:
- Cosine similarity (how similar are the embeddings?)
- L2 distance (Euclidean distance)
- Max absolute difference per dimension
- Mean absolute difference

Usage:
    make test-models
"""

import json
import sys
import time
from typing import Dict, List, Tuple

import requests


def cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot_product = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot_product / (norm_a * norm_b)


def l2_distance(a: List[float], b: List[float]) -> float:
    """Compute L2 (Euclidean) distance."""
    return sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5


def max_abs_diff(a: List[float], b: List[float]) -> float:
    """Maximum absolute difference per dimension."""
    return max(abs(x - y) for x, y in zip(a, b))


def mean_abs_diff(a: List[float], b: List[float]) -> float:
    """Mean absolute difference."""
    return sum(abs(x - y) for x, y in zip(a, b)) / len(a)


def get_embedding(port: int, text: str) -> Tuple[List[float], float]:
    """Get embedding from server, return (embedding, latency_ms)."""
    start = time.time()
    response = requests.post(
        f"http://localhost:{port}/embed",
        json={"input": text},
        headers={"content-type": "application/json"},
        timeout=30,
    )
    response.raise_for_status()
    data = response.json()
    latency = (time.time() - start) * 1000

    if "embedding" not in data:
        if "error" in data:
            raise ValueError(f"Server error on port {port}: {data['error']}")
        else:
            raise ValueError(
                f"Unexpected response format from port {port}. Keys: {list(data.keys())}. Response: {data}"
            )

    return data["embedding"], latency


def compare_models(
    baseline_port: int,
    variant_port: int,
    variant_name: str,
    test_texts: List[str],
) -> Dict:
    """Compare embeddings from baseline vs variant model."""
    results = {
        "variant": variant_name,
        "baseline_port": baseline_port,
        "variant_port": variant_port,
        "texts": [],
        "summary": {},
    }

    all_cosine_sims = []
    all_l2_dists = []
    all_max_diffs = []
    all_mean_diffs = []
    baseline_latencies = []
    variant_latencies = []

    for text in test_texts:
        # Get embeddings from both models
        baseline_emb, baseline_lat = get_embedding(baseline_port, text)
        variant_emb, variant_lat = get_embedding(variant_port, text)

        # Compute metrics
        cosine_sim = cosine_similarity(baseline_emb, variant_emb)
        l2_dist = l2_distance(baseline_emb, variant_emb)
        max_diff = max_abs_diff(baseline_emb, variant_emb)
        mean_diff = mean_abs_diff(baseline_emb, variant_emb)

        all_cosine_sims.append(cosine_sim)
        all_l2_dists.append(l2_dist)
        all_max_diffs.append(max_diff)
        all_mean_diffs.append(mean_diff)
        baseline_latencies.append(baseline_lat)
        variant_latencies.append(variant_lat)

        results["texts"].append(
            {
                "text": text[:50] + ("..." if len(text) > 50 else ""),
                "cosine_similarity": cosine_sim,
                "l2_distance": l2_dist,
                "max_abs_diff": max_diff,
                "mean_abs_diff": mean_diff,
                "baseline_latency_ms": baseline_lat,
                "variant_latency_ms": variant_lat,
            }
        )

    # Summary statistics
    results["summary"] = {
        "cosine_similarity": {
            "mean": sum(all_cosine_sims) / len(all_cosine_sims),
            "min": min(all_cosine_sims),
            "max": max(all_cosine_sims),
        },
        "l2_distance": {
            "mean": sum(all_l2_dists) / len(all_l2_dists),
            "min": min(all_l2_dists),
            "max": max(all_l2_dists),
        },
        "max_abs_diff": {
            "mean": sum(all_max_diffs) / len(all_max_diffs),
            "min": min(all_max_diffs),
            "max": max(all_max_diffs),
        },
        "mean_abs_diff": {
            "mean": sum(all_mean_diffs) / len(all_mean_diffs),
            "min": min(all_mean_diffs),
            "max": max(all_mean_diffs),
        },
        "latency_ms": {
            "baseline_mean": sum(baseline_latencies) / len(baseline_latencies),
            "variant_mean": sum(variant_latencies) / len(variant_latencies),
            "speedup": (
                sum(baseline_latencies) / sum(variant_latencies)
                if sum(variant_latencies) > 0
                else 0
            ),
        },
    }

    return results


def print_results(results: Dict):
    """Pretty print comparison results."""
    print(f"\n{'=' * 80}")
    print(f"Model Variant: {results['variant']}")
    print(f"{'=' * 80}")

    s = results["summary"]

    print(f"\n📊 Similarity Metrics:")
    print(
        f"  Cosine Similarity: {s['cosine_similarity']['mean']:.6f} (min: {s['cosine_similarity']['min']:.6f}, max: {s['cosine_similarity']['max']:.6f})"
    )
    print(
        f"  L2 Distance:       {s['l2_distance']['mean']:.6f} (min: {s['l2_distance']['min']:.6f}, max: {s['l2_distance']['max']:.6f})"
    )
    print(
        f"  Max Abs Diff:      {s['max_abs_diff']['mean']:.6f} (min: {s['max_abs_diff']['min']:.6f}, max: {s['max_abs_diff']['max']:.6f})"
    )
    print(
        f"  Mean Abs Diff:     {s['mean_abs_diff']['mean']:.6f} (min: {s['mean_abs_diff']['min']:.6f}, max: {s['mean_abs_diff']['max']:.6f})"
    )

    print(f"\n⚡ Performance:")
    print(f"  Baseline latency: {s['latency_ms']['baseline_mean']:.2f} ms")
    print(f"  Variant latency:  {s['latency_ms']['variant_mean']:.2f} ms")
    if s["latency_ms"]["speedup"] > 0:
        print(f"  Speedup:          {s['latency_ms']['speedup']:.2f}x")

    print(f"\n📝 Per-Text Details:")
    for i, text_result in enumerate(results["texts"], 1):
        print(f"  [{i}] {text_result['text']}")
        print(
            f"      Cosine: {text_result['cosine_similarity']:.6f}, "
            f"L2: {text_result['l2_distance']:.4f}, "
            f"Max diff: {text_result['max_abs_diff']:.6f}"
        )


def main():
    """Main comparison script."""
    if len(sys.argv) < 3:
        print(
            "Usage: compare_model_variants.py <baseline_port> <variant_port> <variant_name>"
        )
        sys.exit(1)

    baseline_port = int(sys.argv[1])
    variant_port = int(sys.argv[2])
    variant_name = sys.argv[3]

    # Test texts of varying lengths
    test_texts = [
        "ONNX in Rust is fast",
        "Python is also great",
        "Embeddings are useful for semantic search",
        "The quick brown fox jumps over the lazy dog",
        "Machine learning models require careful evaluation",
        "Quantization reduces model size and inference time",
    ]

    print(f"Testing {variant_name} against baseline (fp32, port {baseline_port})")
    print(f"Baseline port: {baseline_port}, Variant port: {variant_port}")
    print(f"Test texts: {len(test_texts)}")

    try:
        results = compare_models(baseline_port, variant_port, variant_name, test_texts)
        print_results(results)
        return results
    except requests.exceptions.RequestException as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        print("Make sure servers are running on the specified ports.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
