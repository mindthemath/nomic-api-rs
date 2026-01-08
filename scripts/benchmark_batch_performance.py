#!/usr/bin/env python3
"""
Comprehensive batch size performance benchmark for nomic-serve API.

Tests different batch sizes to find the optimal throughput sweet spot.
Measures latency percentiles, throughput, and CPU utilization.
"""

import argparse
import asyncio
import base64
import io
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import aiohttp
from PIL import Image


def create_test_image(seed: int = 42) -> str:
    """Create a test image and return as base64."""
    import numpy as np

    np.random.seed(seed)
    pixels = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    img = Image.fromarray(pixels, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{img_base64}"


async def benchmark_batch_size(
    session: aiohttp.ClientSession,
    base_url: str,
    endpoint: str,
    batch_size: int,
    num_runs: int = 20,
    warmup: int = 3,
) -> Dict:
    """Benchmark a specific batch size."""
    url = f"{base_url}/{endpoint}/batch"

    # Prepare payload
    if endpoint == "txt":
        payload = {
            "inputs": [
                f"Test text {i} for batch size {batch_size}" for i in range(batch_size)
            ],
            "dim": 768,
            "prefix": "search_query",
        }
    else:  # img
        test_image = create_test_image()
        payload = {
            "contents": [test_image] * batch_size,
            "dim": 768,
        }

    # Warmup
    for _ in range(warmup):
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status != 200:
                    return {"error": f"HTTP {resp.status}"}
                await resp.json()
        except Exception as e:
            return {"error": str(e)}

    # Benchmark
    latencies = []
    total_tokens_list = []  # For text endpoints
    errors = 0

    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            async with session.post(
                url, json=payload, timeout=aiohttp.ClientTimeout(total=300)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    elapsed = (time.perf_counter() - start) * 1000  # ms
                    latencies.append(elapsed)

                    # Track tokens for text endpoints
                    if endpoint == "txt" and "tokens" in data:
                        tokens = data["tokens"]
                        if isinstance(tokens, list):
                            total_tokens = sum(tokens)
                        else:
                            total_tokens = tokens
                        total_tokens_list.append(total_tokens)
                else:
                    errors += 1
                    error_text = await resp.text()
                    if errors == 1:  # Only print first error to avoid spam
                        try:
                            error_json = json.loads(error_text)
                            error_msg = error_json.get("error", error_text)
                        except:
                            error_msg = error_text
                        print(f"  ⚠️  Error: HTTP {resp.status} - {error_msg[:200]}")
        except asyncio.TimeoutError:
            errors += 1
            print(f"  ⚠️  Timeout for batch_size={batch_size}")
        except Exception as e:
            errors += 1
            print(f"  ⚠️  Exception: {e}")

    if not latencies:
        return {"error": "All requests failed"}

    latencies.sort()
    n = len(latencies)
    avg_latency = statistics.mean(latencies)

    # Calculate throughput
    throughput_rps = (1000.0 / avg_latency) if latencies else 0
    throughput_items_per_sec = (batch_size * 1000.0 / avg_latency) if latencies else 0

    # Calculate tokens/sec for text endpoints
    throughput_tokens_per_sec = None
    avg_tokens_per_batch = None
    if endpoint == "txt" and total_tokens_list:
        avg_tokens_per_batch = statistics.mean(total_tokens_list)
        throughput_tokens_per_sec = (
            (avg_tokens_per_batch * 1000.0 / avg_latency) if avg_latency > 0 else 0
        )

    return {
        "batch_size": batch_size,
        "runs": num_runs,
        "successful": len(latencies),
        "errors": errors,
        "latency_total_avg": avg_latency,  # Total batch latency
        "latency_per_item": avg_latency / batch_size,  # Per-item latency
        "latency_std": statistics.stdev(latencies) if len(latencies) > 1 else 0,
        "latency_min": min(latencies),
        "latency_max": max(latencies),
        "latency_p50": latencies[n // 2],
        "latency_p95": latencies[int(n * 0.95)] if n > 1 else latencies[0],
        "latency_p99": latencies[int(n * 0.99)] if n > 1 else latencies[0],
        "throughput_rps": throughput_rps,
        "throughput_items_per_sec": throughput_items_per_sec,
        "throughput_tokens_per_sec": throughput_tokens_per_sec,
        "avg_tokens_per_batch": avg_tokens_per_batch,
    }


async def run_benchmark_suite(
    base_url: str,
    endpoint: str,
    batch_sizes: List[int],
    num_runs: int = 20,
    warmup: int = 3,
    early_stop: bool = True,
    degradation_threshold: float = 0.20,
) -> List[Dict]:
    """Run benchmark suite for multiple batch sizes."""
    print(f"\n{'=' * 80}")
    print(f"Benchmarking /{endpoint}/batch")
    print(f"{'=' * 80}")
    print(f"Batch sizes to test: {batch_sizes}")
    print(f"Runs per batch size: {num_runs}")
    print(f"Warmup runs: {warmup}")
    if early_stop:
        print(
            f"Early stopping: enabled (stops if {degradation_threshold*100:.0f}% worse than best)"
        )
    else:
        print(f"Early stopping: disabled (will test all batch sizes)")

    results = []
    best_per_item_latency = None
    best_throughput = None
    best_batch_size = None
    async with aiohttp.ClientSession() as session:
        for batch_size in batch_sizes:
            print(f"\nTesting batch_size={batch_size}...", end="", flush=True)
            result = await benchmark_batch_size(
                session, base_url, endpoint, batch_size, num_runs, warmup
            )
            if "error" in result:
                print(f" ❌ {result['error']}")
                continue
            results.append(result)

            per_item_latency = result["latency_per_item"]
            total_latency = result["latency_total_avg"]

            # Format throughput based on endpoint
            if endpoint == "txt" and result.get("throughput_tokens_per_sec"):
                throughput_str = f"{result['throughput_tokens_per_sec']:.2f} tok/s"
                if result.get("avg_tokens_per_batch"):
                    throughput_str += f" ({result['throughput_items_per_sec']:.2f} items/s, ~{result['avg_tokens_per_batch']:.0f} tok/batch)"
                throughput_val = result["throughput_tokens_per_sec"]
            else:
                throughput_str = f"{result['throughput_items_per_sec']:.2f} items/s"
                throughput_val = result["throughput_items_per_sec"]

            print(f" ✓ ({result['successful']}/{result['runs']} successful)")
            print(
                f"   Total batch latency: {total_latency:.2f}ms | Per-item: {per_item_latency:.2f}ms/item | Throughput: {throughput_str}"
            )

            # Early stopping logic
            if early_stop and len(results) > 1:
                if best_per_item_latency is None:
                    best_per_item_latency = per_item_latency
                    best_throughput = throughput_val
                    best_batch_size = batch_size
                else:
                    # Check if this is better
                    if per_item_latency < best_per_item_latency:
                        improvement = (
                            (best_per_item_latency - per_item_latency)
                            / best_per_item_latency
                        ) * 100
                        best_per_item_latency = per_item_latency
                        best_throughput = throughput_val
                        best_batch_size = batch_size
                        print(
                            f"   🎯 New best! {improvement:.1f}% better per-item latency"
                        )
                    # Check if significantly worse
                    elif per_item_latency > best_per_item_latency * (
                        1 + degradation_threshold
                    ):
                        degradation = (
                            per_item_latency / best_per_item_latency - 1
                        ) * 100
                        print(
                            f"   ⚠️  Performance degraded: {degradation:.1f}% worse than best"
                        )
                        print(
                            f"   🛑 Stopping early (best was batch_size={best_batch_size} with {best_per_item_latency:.2f}ms/item)"
                        )
                        break

    if early_stop and best_batch_size and len(results) > 1:
        print(f"\n✓ Early stopping: Best performance at batch_size={best_batch_size}")
        print(f"  Per-item latency: {best_per_item_latency:.2f}ms/item")
        if endpoint == "txt" and results[0].get("throughput_tokens_per_sec"):
            print(f"  Throughput: {best_throughput:.2f} tok/s")
        else:
            print(f"  Throughput: {best_throughput:.2f} items/s")

    return results


def print_results_table(results: List[Dict], endpoint: str):
    """Print formatted results table."""
    if not results:
        print("\nNo results to display")
        return

    print(f"\n{'=' * 120}")
    print(f"RESULTS: /{endpoint}/batch")
    print(f"{'=' * 120}")

    # Header - adjust based on endpoint
    if endpoint == "txt" and results and results[0].get("throughput_tokens_per_sec"):
        header = (
            f"{'Batch':<8} "
            f"{'Total Batch Latency (ms)':<35} "
            f"{'Per-Item Latency (ms)':<20} "
            f"{'Throughput':<30} "
            f"{'Success':<10} "
            f"{'Speedup':<10}"
        )
    else:
        header = (
            f"{'Batch':<8} "
            f"{'Total Batch Latency (ms)':<35} "
            f"{'Per-Item Latency (ms)':<20} "
            f"{'Throughput':<30} "
            f"{'Success':<10} "
            f"{'Speedup':<10}"
        )
    print(header)
    print("-" * 120)

    # Find baseline (batch_size=1)
    baseline = next((r for r in results if r["batch_size"] == 1), None)
    baseline_latency = baseline["latency_total_avg"] if baseline else None

    # Rows
    for result in results:
        batch_size = result["batch_size"]
        latency_str = (
            f"avg={result['latency_total_avg']:.1f}±{result['latency_std']:.1f} "
            f"p50={result['latency_p50']:.1f} p95={result['latency_p95']:.1f}"
        )
        per_item_str = f"{result['latency_per_item']:.2f}"

        # Format throughput based on endpoint
        if endpoint == "txt" and result.get("throughput_tokens_per_sec"):
            throughput_str = (
                f"{result['throughput_tokens_per_sec']:.1f} tok/s "
                f"({result['throughput_items_per_sec']:.1f} items/s)"
            )
        else:
            throughput_str = (
                f"{result['throughput_items_per_sec']:.1f} items/s "
                f"({result['throughput_rps']:.1f} req/s)"
            )

        success_str = f"{result['successful']}/{result['runs']}"

        if baseline and batch_size > 1:
            speedup = (baseline_latency * batch_size) / result["latency_total_avg"]
            speedup_str = f"{speedup:.2f}x"
        else:
            speedup_str = "1.00x"

        row = (
            f"{batch_size:<8} "
            f"{latency_str:<35} "
            f"{per_item_str:<20} "
            f"{throughput_str:<30} "
            f"{success_str:<10} "
            f"{speedup_str:<10}"
        )
        print(row)

    # Summary
    print(f"\n{'=' * 120}")
    print("SUMMARY")
    print(f"{'=' * 120}")

    if len(results) > 1:
        # For text, prefer tokens/sec; for images, use items/sec
        if endpoint == "txt" and results[0].get("throughput_tokens_per_sec"):
            best_throughput = max(
                results, key=lambda x: x.get("throughput_tokens_per_sec", 0) or 0
            )
            throughput_val = best_throughput.get("throughput_tokens_per_sec", 0)
            throughput_unit = "tok/s"
            throughput_alt = (
                f" ({best_throughput['throughput_items_per_sec']:.2f} items/s)"
            )
        else:
            best_throughput = max(results, key=lambda x: x["throughput_items_per_sec"])
            throughput_val = best_throughput["throughput_items_per_sec"]
            throughput_unit = "items/s"
            throughput_alt = ""

        best_latency = min(results, key=lambda x: x["latency_per_item"])

        print(f"\nBest throughput: batch_size={best_throughput['batch_size']}")
        print(f"  {throughput_val:.2f} {throughput_unit}{throughput_alt}")
        print(f"  Total batch latency: {best_throughput['latency_total_avg']:.2f}ms")
        print(f"  Per-item latency: {best_throughput['latency_per_item']:.2f}ms/item")

        print(f"\nBest per-item latency: batch_size={best_latency['batch_size']}")
        print(f"  {best_latency['latency_per_item']:.2f}ms per item")
        if endpoint == "txt" and best_latency.get("throughput_tokens_per_sec"):
            print(
                f"  {best_latency['throughput_tokens_per_sec']:.2f} tok/s ({best_latency['throughput_items_per_sec']:.2f} items/s)"
            )
        else:
            print(f"  {best_latency['throughput_items_per_sec']:.2f} items/s")


def save_markdown(results: List[Dict], endpoint: str, output_file: str):
    """Save results to markdown file."""
    if not results:
        return

    with open(output_file, "w") as f:
        f.write(f"# Batch Size Performance Benchmark: /{endpoint}/batch\n\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Results\n\n")
        f.write(
            "| Batch | Avg Latency (ms) | Std Dev | P50 | P95 | P99 | Throughput (items/s) | RPS | Success | Speedup |\n"
        )
        f.write(
            "|-------|-----------------|---------|-----|-----|-----|---------------------|-----|---------|--------|\n"
        )

        baseline = next((r for r in results if r["batch_size"] == 1), None)
        baseline_latency = baseline["latency_avg"] if baseline else None

        for result in results:
            batch_size = result["batch_size"]
            if baseline and batch_size > 1:
                speedup = (baseline_latency * batch_size) / result["latency_avg"]
            else:
                speedup = 1.0

            f.write(
                f"| {batch_size} | "
                f"{result['latency_avg']:.2f} | "
                f"{result['latency_std']:.2f} | "
                f"{result['latency_p50']:.2f} | "
                f"{result['latency_p95']:.2f} | "
                f"{result['latency_p99']:.2f} | "
                f"{result['throughput_items_per_sec']:.2f} | "
                f"{result['throughput_rps']:.2f} | "
                f"{result['successful']}/{result['runs']} | "
                f"{speedup:.2f}x |\n"
            )

        if len(results) > 1:
            if endpoint == "txt" and results[0].get("throughput_tokens_per_sec"):
                best_throughput = max(
                    results, key=lambda x: x.get("throughput_tokens_per_sec", 0) or 0
                )
                throughput_val = best_throughput.get("throughput_tokens_per_sec", 0)
                throughput_unit = "tok/s"
            else:
                best_throughput = max(
                    results, key=lambda x: x["throughput_items_per_sec"]
                )
                throughput_val = best_throughput["throughput_items_per_sec"]
                throughput_unit = "items/s"

            best_latency = min(results, key=lambda x: x["latency_per_item"])

            f.write("\n## Summary\n\n")
            f.write(
                f"- **Best throughput**: batch_size={best_throughput['batch_size']} "
                f"({throughput_val:.2f} {throughput_unit})\n"
            )
            f.write(
                f"- **Best per-item latency**: batch_size={best_latency['batch_size']} "
                f"({best_latency['latency_per_item']:.2f}ms per item)\n"
            )

    print(f"\n✓ Results saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark batch size performance")
    parser.add_argument(
        "--url",
        type=str,
        default="http://localhost:8080",
        help="Base URL of the API server",
    )
    parser.add_argument(
        "--endpoint",
        type=str,
        choices=["txt", "img", "both"],
        default="both",
        help="Endpoint to benchmark",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=2056,
        help="Maximum batch size to test (default: 2056)",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=20,
        help="Number of runs per batch size (default: 20)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=3,
        help="Number of warmup runs (default: 3)",
    )
    parser.add_argument(
        "--markdown",
        type=str,
        help="Save results to markdown file",
    )
    parser.add_argument(
        "--no-early-stop",
        action="store_true",
        help="Disable early stopping when performance degrades",
    )
    parser.add_argument(
        "--degradation-threshold",
        type=float,
        default=0.20,
        help="Stop early if per-item latency is this fraction worse than best (default: 0.20 = 20%%)",
    )
    args = parser.parse_args()

    # Determine batch sizes to test
    # Start with powers of 2 up to 1024, then test the max, then double if needed
    batch_sizes = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    if args.max_batch_size > 1024:
        # Add intermediate sizes between 1024 and max_batch_size
        if args.max_batch_size <= 2048:
            batch_sizes.extend([1536, args.max_batch_size])
        else:
            # Add some intermediate points
            step = (args.max_batch_size - 1024) // 3
            batch_sizes.extend([1024 + step, 1024 + 2 * step, args.max_batch_size])

        # Test double the max if we're still improving
        batch_sizes.append(args.max_batch_size * 2)

    # Remove duplicates and sort, filter to max
    batch_sizes = sorted(
        list(set([b for b in batch_sizes if b <= args.max_batch_size * 2]))
    )

    print("=" * 80)
    print("Batch Size Performance Benchmark")
    print("=" * 80)
    print(f"\nServer: {args.url}")
    print(f"Endpoint: {args.endpoint}")
    print(f"Batch sizes: {batch_sizes}")
    print(f"Runs per batch size: {args.runs}")
    print(f"Warmup runs: {args.warmup}")

    # Check server is running and verify model configuration
    try:
        import requests

        resp = requests.get(f"{args.url}/health", timeout=2)
        if resp.status_code != 200:
            print(f"\n❌ Server health check failed: HTTP {resp.status_code}")
            sys.exit(1)

        # Check model info
        info_resp = requests.get(f"{args.url}/info", timeout=2)
        if info_resp.status_code == 200:
            info = info_resp.json()
            txt_model = info.get("txt_model", "")
            txt_max = info.get("txt_max_batch_size")
            img_model = info.get("img_model", "")
            img_max = info.get("img_max_batch_size")

            print(f"\nServer configuration:")
            if txt_model:
                print(f"  Text model: {txt_model}")
                print(f"  Text max batch size: {txt_max}")
                if "quantized" in txt_model.lower() and args.endpoint in [
                    "txt",
                    "both",
                ]:
                    print(
                        f"  ⚠️  WARNING: Quantized text model detected - batching disabled!"
                    )
                    print(
                        f"     Use FP32 model for batching: TXT_MODEL=models/txt/model.onnx make run-benchmark"
                    )
            if img_model:
                print(f"  Image model: {img_model}")
                print(f"  Image max batch size: {img_max}")
    except Exception as e:
        print(f"\n❌ Cannot connect to server at {args.url}: {e}")
        print("   Make sure the server is running: make run-benchmark")
        sys.exit(1)

    print("✓ Server is running")

    # Run benchmarks
    all_results = {}
    early_stop = not args.no_early_stop

    if args.endpoint in ["txt", "both"]:
        txt_results = asyncio.run(
            run_benchmark_suite(
                args.url,
                "txt",
                batch_sizes,
                args.runs,
                args.warmup,
                early_stop=early_stop,
                degradation_threshold=args.degradation_threshold,
            )
        )
        if txt_results:
            print_results_table(txt_results, "txt")
            all_results["txt"] = txt_results
            if args.markdown:
                save_markdown(
                    txt_results, "txt", args.markdown.replace(".md", "_txt.md")
                )

    if args.endpoint in ["img", "both"]:
        img_results = asyncio.run(
            run_benchmark_suite(
                args.url,
                "img",
                batch_sizes,
                args.runs,
                args.warmup,
                early_stop=early_stop,
                degradation_threshold=args.degradation_threshold,
            )
        )
        if img_results:
            print_results_table(img_results, "img")
            all_results["img"] = img_results
            if args.markdown:
                save_markdown(
                    img_results, "img", args.markdown.replace(".md", "_img.md")
                )

    # Final summary
    print(f"\n{'=' * 80}")
    print("BENCHMARK COMPLETE")
    print(f"{'=' * 80}")

    if all_results:
        print("\nRecommendations:")
        for endpoint, results in all_results.items():
            if results:
                if endpoint == "txt" and results[0].get("throughput_tokens_per_sec"):
                    best = max(
                        results,
                        key=lambda x: x.get("throughput_tokens_per_sec", 0) or 0,
                    )
                    throughput_str = (
                        f"{best.get('throughput_tokens_per_sec', 0):.1f} tok/s"
                    )
                else:
                    best = max(results, key=lambda x: x["throughput_items_per_sec"])
                    throughput_str = f"{best['throughput_items_per_sec']:.1f} items/s"
                print(
                    f"  /{endpoint}/batch: Optimal batch_size={best['batch_size']} ({throughput_str})"
                )


if __name__ == "__main__":
    main()
