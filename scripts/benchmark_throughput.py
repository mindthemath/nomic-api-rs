#!/usr/bin/env python3
"""
Benchmark API server throughput (requests per second).

Tests both text and image endpoints with concurrent requests.

Usage:
    # Start server first: make run
    python scripts/benchmark_throughput.py [--endpoint txt|img|both] [--concurrent 10] [--requests 100]
"""

import argparse
import asyncio
import time
from typing import List

import aiohttp


async def make_request(
    session: aiohttp.ClientSession, url: str, payload: dict
) -> tuple[float, bool]:
    """Make a single request and return (latency_ms, success)."""
    start = time.perf_counter()
    try:
        async with session.post(
            url, json=payload, timeout=aiohttp.ClientTimeout(total=30)
        ) as resp:
            if resp.status == 200:
                await resp.json()  # Read response
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed, True
            else:
                elapsed = (time.perf_counter() - start) * 1000
                return elapsed, False
    except Exception as e:
        elapsed = (time.perf_counter() - start) * 1000
        print(f"Request failed: {e}")
        return elapsed, False


async def benchmark_endpoint(
    base_url: str,
    endpoint: str,
    payload: dict,
    concurrent: int,
    total_requests: int,
) -> dict:
    """Benchmark an endpoint with concurrent requests."""
    url = f"{base_url}/{endpoint}"

    # Create semaphore to limit concurrency
    semaphore = asyncio.Semaphore(concurrent)

    async def bounded_request(session, url, payload):
        async with semaphore:
            return await make_request(session, url, payload)

    async with aiohttp.ClientSession() as session:
        # Create all tasks
        tasks = [bounded_request(session, url, payload) for _ in range(total_requests)]

        start = time.perf_counter()
        results = await asyncio.gather(*tasks)
        total_time = time.perf_counter() - start

    latencies = [r[0] for r in results]
    successes = [r[1] for r in results]

    success_count = sum(successes)
    success_rate = (success_count / total_requests) * 100

    return {
        "endpoint": endpoint,
        "total_requests": total_requests,
        "successful": success_count,
        "failed": total_requests - success_count,
        "success_rate": success_rate,
        "total_time_sec": total_time,
        "rps": total_requests / total_time,
        "avg_latency_ms": sum(latencies) / len(latencies) if latencies else 0,
        "p50_latency_ms": sorted(latencies)[len(latencies) // 2] if latencies else 0,
        "p95_latency_ms": (
            sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0
        ),
        "p99_latency_ms": (
            sorted(latencies)[int(len(latencies) * 0.99)] if latencies else 0
        ),
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark API server throughput")
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
        "--concurrent",
        type=int,
        default=10,
        help="Number of concurrent requests",
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=100,
        help="Total number of requests to make",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("API Server Throughput Benchmark")
    print("=" * 70)
    print(f"\nServer: {args.url}")
    print(f"Concurrent requests: {args.concurrent}")
    print(f"Total requests: {args.requests}")

    # Test payloads
    txt_payload = {"input": "This is a test sentence for embedding generation."}
    img_payload = {"content": "https://picsum.photos/400/300"}

    results = []

    async def run_benchmarks():
        if args.endpoint in ["txt", "both"]:
            print(f"\n{'=' * 70}")
            print("Benchmarking /txt/embed")
            print(f"{'=' * 70}")
            result = await benchmark_endpoint(
                args.url, "txt/embed", txt_payload, args.concurrent, args.requests
            )
            results.append(result)

        if args.endpoint in ["img", "both"]:
            print(f"\n{'=' * 70}")
            print("Benchmarking /img/embed")
            print(f"{'=' * 70}")
            result = await benchmark_endpoint(
                args.url, "img/embed", img_payload, args.concurrent, args.requests
            )
            results.append(result)

    asyncio.run(run_benchmarks())

    # Print results
    print(f"\n{'=' * 70}")
    print("RESULTS")
    print(f"{'=' * 70}")

    for result in results:
        print(f"\n{result['endpoint']}:")
        print(
            f"  Requests: {result['successful']}/{result['total_requests']} successful ({result['success_rate']:.1f}%)"
        )
        print(f"  Throughput: {result['rps']:.2f} requests/second")
        print(f"  Latency:")
        print(f"    Average: {result['avg_latency_ms']:.2f} ms")
        print(f"    P50:     {result['p50_latency_ms']:.2f} ms")
        print(f"    P95:     {result['p95_latency_ms']:.2f} ms")
        print(f"    P99:     {result['p99_latency_ms']:.2f} ms")

    # Recommendations
    print(f"\n{'=' * 70}")
    print("RECOMMENDATIONS")
    print(f"{'=' * 70}")

    for result in results:
        rps = result["rps"]
        p95 = result["p95_latency_ms"]

        print(f"\n{result['endpoint']}:")
        if rps < 10:
            print(f"  ⚠️  Low throughput ({rps:.1f} RPS)")
            print(f"     - Consider horizontal scaling (multiple server instances)")
            print(f"     - Use load balancer (nginx, HAProxy)")
            print(f"     - Increase concurrent request handling")
        elif rps < 50:
            print(f"  ✓ Moderate throughput ({rps:.1f} RPS)")
            print(f"     - Can improve with horizontal scaling")
        else:
            print(f"  ✓ Good throughput ({rps:.1f} RPS)")

        if p95 > 1000:
            print(f"  ⚠️  High P95 latency ({p95:.0f} ms)")
            print(f"     - Consider optimizing preprocessing")
            print(f"     - Check if model inference is bottleneck")


if __name__ == "__main__":
    main()
