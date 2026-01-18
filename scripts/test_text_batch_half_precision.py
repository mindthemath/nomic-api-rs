#!/usr/bin/env python3
"""
Test text model batching with PyTorch/transformers using half-precision (FP16/BF16).

Compares FP32 vs FP16 vs BF16 to see if half-precision causes batching interference
similar to ONNX quantized model.
"""

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

try:
    from transformers import AutoModel, AutoTokenizer
except ImportError:
    print("Missing: pip install transformers torch")
    sys.exit(1)


def mean_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    return torch.sum(embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def embed_single(tokenizer, model, text: str, device: str) -> torch.Tensor:
    """Embed a single text (batch_size=1)."""
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        embedding = mean_pool(last_hidden, inputs["attention_mask"])
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding[0]


def embed_batch(tokenizer, model, texts: list, device: str) -> list[torch.Tensor]:
    """Embed multiple texts in a single batch."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        embeddings = mean_pool(last_hidden, inputs["attention_mask"])
        embeddings = F.normalize(embeddings, p=2, dim=1)

    return [embeddings[i] for i in range(len(texts))]


def test_precision(precision: str, model, tokenizer, device: str):
    """Test batching with a specific precision."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {precision}")
    print(f"{'=' * 70}")

    # Set model to appropriate precision
    if precision == "FP32":
        model_fp = model.float()
    elif precision == "FP16":
        if not torch.cuda.is_available():
            print(f"⚠️  FP16 requires CUDA, skipping...")
            return None
        model_fp = model.half()
    elif precision == "BF16":
        if not torch.cuda.is_available():
            print(f"⚠️  BF16 requires CUDA, skipping...")
            return None
        model_fp = model.bfloat16()
    else:
        raise ValueError(f"Unknown precision: {precision}")

    model_fp = model_fp.to(device)
    model_fp.eval()

    # Test texts
    texts = [
        "ONNX in Rust is fast",
        "Python is also great",
        "Embeddings are useful",
        "Hello world",
    ]

    print(f"\nTest texts:")
    for i, text in enumerate(texts):
        tokens = tokenizer.encode(text)
        print(f"  [{i}] {len(tokens)} tokens: {text}")

    # Test 1: Same text, single vs batched
    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)

    ref_emb = embed_single(tokenizer, model_fp, texts[0], device)
    batch_embs = embed_batch(tokenizer, model_fp, [texts[0], texts[0]], device)

    diff_0 = torch.abs(ref_emb - batch_embs[0]).max().item()
    diff_1 = torch.abs(ref_emb - batch_embs[1]).max().item()
    diff_batch = torch.abs(batch_embs[0] - batch_embs[1]).max().item()

    status_0 = "✓ identical" if diff_0 < 0.0001 else "✗ DIFFERENT"
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_batch = "✓ identical" if diff_batch < 0.0001 else "✗ DIFFERENT"
    print(f"{'Same text: single vs batch[0]':<40} {diff_0:.6f} {status_0}")
    print(f"{'Same text: single vs batch[1]':<40} {diff_1:.6f} {status_1}")
    print(f"{'Same text: batch[0] vs batch[1]':<40} {diff_batch:.6f} {status_batch}")

    # Test 2: Different texts in batch
    single_emb1 = embed_single(tokenizer, model_fp, texts[0], device)
    single_emb2 = embed_single(tokenizer, model_fp, texts[1], device)
    batch_embs = embed_batch(tokenizer, model_fp, [texts[0], texts[1]], device)

    diff_1 = torch.abs(single_emb1 - batch_embs[0]).max().item()
    diff_2 = torch.abs(single_emb2 - batch_embs[1]).max().item()
    cos_sim_1 = torch.dot(single_emb1, batch_embs[0]).item()
    cos_sim_2 = torch.dot(single_emb2, batch_embs[1]).item()

    print(f"\n{'Test Case':<40} {'Max Diff':<15} {'Cosine Sim'}")
    print("-" * 70)
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_2 = "✓ identical" if diff_2 < 0.0001 else "✗ DIFFERENT"
    print(
        f"{'Different texts: text[0] single vs batch[0]':<40} {diff_1:<15.6f} {cos_sim_1:.6f} {status_1}"
    )
    print(
        f"{'Different texts: text[1] single vs batch[1]':<40} {diff_2:<15.6f} {cos_sim_2:.6f} {status_2}"
    )

    # Test 3: Larger batch
    single_embs = [embed_single(tokenizer, model_fp, text, device) for text in texts]
    batch_embs = embed_batch(tokenizer, model_fp, texts, device)

    max_diff = max(
        torch.abs(single_embs[i] - batch_embs[i]).max().item() for i in range(4)
    )
    avg_cos_sim = (
        sum(torch.dot(single_embs[i], batch_embs[i]).item() for i in range(4)) / 4
    )

    print(f"\nLarger batch (4 texts):")
    print(f"  Max difference: {max_diff:.6f}")
    print(f"  Average cosine similarity: {avg_cos_sim:.6f}")
    status = "✓ identical" if max_diff < 0.0001 else "✗ DIFFERENT"
    print(f"  Result: {status}")

    return {
        "precision": precision,
        "same_text_diff": diff_0,
        "different_text_diff_1": diff_1,
        "different_text_diff_2": diff_2,
        "different_text_cos_1": cos_sim_1,
        "different_text_cos_2": cos_sim_2,
        "large_batch_diff": max_diff,
        "large_batch_cos": avg_cos_sim,
    }


def main():
    print("=" * 70)
    print("Text Model Batching: Half-Precision (FP16/BF16) Test")
    print("Model: nomic-embed-text-v1.5")
    print("=" * 70)

    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    print("(This may take a minute on first run - downloads from HuggingFace)")

    try:
        tokenizer = AutoTokenizer.from_pretrained("nomic-ai/nomic-embed-text-v1.5")
        model = AutoModel.from_pretrained(
            "nomic-ai/nomic-embed-text-v1.5", trust_remote_code=True
        )
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        sys.exit(1)

    # Check device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    if device == "cpu":
        print("\n⚠️  Warning: Half-precision (FP16/BF16) requires CUDA")
        print("   Will only test FP32 on CPU")
        precisions = ["FP32"]
    else:
        precisions = ["FP32", "FP16", "BF16"]

    results = {}

    # Test each precision
    for precision in precisions:
        result = test_precision(precision, model, tokenizer, device)
        if result:
            results[precision] = result

    # Compare results
    if len(results) > 1:
        print(f"\n{'=' * 70}")
        print("COMPARISON: FP32 vs Half-Precision")
        print(f"{'=' * 70}")

        fp32 = results.get("FP32")
        fp16 = results.get("FP16")
        bf16 = results.get("BF16")

        print(f"\n{'Metric':<40} {'FP32':<15} {'FP16':<15} {'BF16':<15}")
        print("-" * 85)

        if fp32 and fp16:
            print(
                f"{'Different texts diff (text 1)':<40} {fp32['different_text_diff_1']:<15.6f} {fp16['different_text_diff_1']:<15.6f} {'N/A':<15}"
            )
            print(
                f"{'Different texts diff (text 2)':<40} {fp32['different_text_diff_2']:<15.6f} {fp16['different_text_diff_2']:<15.6f} {'N/A':<15}"
            )
            print(
                f"{'Different texts cos sim (text 1)':<40} {fp32['different_text_cos_1']:<15.6f} {fp16['different_text_cos_1']:<15.6f} {'N/A':<15}"
            )
            print(
                f"{'Different texts cos sim (text 2)':<40} {fp32['different_text_cos_2']:<15.6f} {fp16['different_text_cos_2']:<15.6f} {'N/A':<15}"
            )
            print(
                f"{'Large batch max diff':<40} {fp32['large_batch_diff']:<15.6f} {fp16['large_batch_diff']:<15.6f} {'N/A':<15}"
            )
            print(
                f"{'Large batch cos sim':<40} {fp32['large_batch_cos']:<15.6f} {fp16['large_batch_cos']:<15.6f} {'N/A':<15}"
            )

        if fp32 and bf16:
            print(f"\n{'Metric':<40} {'FP32':<15} {'BF16':<15}")
            print("-" * 70)
            print(
                f"{'Different texts diff (text 1)':<40} {fp32['different_text_diff_1']:<15.6f} {bf16['different_text_diff_1']:<15.6f}"
            )
            print(
                f"{'Different texts diff (text 2)':<40} {fp32['different_text_diff_2']:<15.6f} {bf16['different_text_diff_2']:<15.6f}"
            )
            print(
                f"{'Different texts cos sim (text 1)':<40} {fp32['different_text_cos_1']:<15.6f} {bf16['different_text_cos_1']:<15.6f}"
            )
            print(
                f"{'Different texts cos sim (text 2)':<40} {fp32['different_text_cos_2']:<15.6f} {bf16['different_text_cos_2']:<15.6f}"
            )
            print(
                f"{'Large batch max diff':<40} {fp32['large_batch_diff']:<15.6f} {bf16['large_batch_diff']:<15.6f}"
            )
            print(
                f"{'Large batch cos sim':<40} {fp32['large_batch_cos']:<15.6f} {bf16['large_batch_cos']:<15.6f}"
            )

    # Compare with ONNX results
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH ONNX RESULTS")
    print(f"{'=' * 70}")
    print(f"\nONNX quantized (INT8):")
    print(f"  - Different texts: ~0.5-0.6 max diff (severe interference)")
    print(f"  - Cosine similarity: ~50-60% (unusable)")
    print(f"\nONNX FP32:")
    print(f"  - Different texts: 0.000000 max diff (perfect)")
    print(f"  - Cosine similarity: 100% (perfect)")

    if len(results) > 0:
        fp32_result = results.get("FP32")
        fp16_result = results.get("FP16")
        bf16_result = results.get("BF16")

        print(f"\nPyTorch results:")
        if fp32_result:
            print(
                f"  FP32: {fp32_result['different_text_diff_1']:.6f} max diff, {fp32_result['different_text_cos_1']:.6f} cos sim"
            )
        if fp16_result:
            print(
                f"  FP16: {fp16_result['different_text_diff_1']:.6f} max diff, {fp16_result['different_text_cos_1']:.6f} cos sim"
            )
        if bf16_result:
            print(
                f"  BF16: {bf16_result['different_text_diff_1']:.6f} max diff, {bf16_result['different_text_cos_1']:.6f} cos sim"
            )

    # Conclusion
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")

    if fp16_result and fp16_result["different_text_diff_1"] < 0.0001:
        print("\n✅ FP16 batches PERFECTLY - no interference!")
        print(
            "   Half-precision does NOT cause the same issues as ONNX INT8 quantization."
        )
    elif fp16_result and fp16_result["different_text_diff_1"] < 0.01:
        print("\n⚠️  FP16 shows MINIMAL interference (~0.01)")
        print("   Much better than ONNX quantized (~0.5), but not perfect like FP32.")
    elif fp16_result:
        print("\n❌ FP16 shows SIGNIFICANT interference")
        print("   Similar to ONNX quantized - half-precision may cause issues.")
    else:
        print("\n⚠️  Could not test FP16 (requires CUDA)")


if __name__ == "__main__":
    main()
