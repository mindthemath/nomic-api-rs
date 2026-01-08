#!/usr/bin/env python3
"""
Test text model batching with PyTorch/transformers implementation.

Verifies if PyTorch batches correctly (like vision model does).
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
        # Get last hidden state
        last_hidden = outputs.last_hidden_state
        # Mean pool over non-padding tokens
        embedding = mean_pool(last_hidden, inputs["attention_mask"])
        # L2 normalize
        embedding = F.normalize(embedding, p=2, dim=1)

    return embedding[0]  # Return [768] tensor


def embed_batch(tokenizer, model, texts: list, device: str) -> list[torch.Tensor]:
    """Embed multiple texts in a single batch."""
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        last_hidden = outputs.last_hidden_state
        # Mean pool over non-padding tokens for each text
        embeddings = mean_pool(last_hidden, inputs["attention_mask"])
        # L2 normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)

    # Return list of [768] tensors
    return [embeddings[i] for i in range(len(texts))]


def main():
    print("=" * 70)
    print("Text Model Batching Test (Transformers/PyTorch)")
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
    model = model.to(device)
    print(f"Using device: {device}")

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
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 1: Same text, single vs batched")
    print(f"{'=' * 70}")

    ref_emb = embed_single(tokenizer, model, texts[0], device)
    print(f"Reference (text[0] alone): {ref_emb[:3].cpu().numpy().round(4)}")

    # Batch: same text × 2
    batch_embs = embed_batch(tokenizer, model, [texts[0], texts[0]], device)

    diff_0 = torch.abs(ref_emb - batch_embs[0]).max().item()
    diff_1 = torch.abs(ref_emb - batch_embs[1]).max().item()
    diff_batch = torch.abs(batch_embs[0] - batch_embs[1]).max().item()

    print(f"\n{'Test Case':<40} {'Max Diff'}")
    print("-" * 50)
    status_0 = "✓ identical" if diff_0 < 0.0001 else "✗ DIFFERENT"
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_batch = "✓ identical" if diff_batch < 0.0001 else "✗ DIFFERENT"
    print(f"{'Single vs batch[0]':<40} {diff_0:.6f} {status_0}")
    print(f"{'Single vs batch[1]':<40} {diff_1:.6f} {status_1}")
    print(f"{'batch[0] vs batch[1]':<40} {diff_batch:.6f} {status_batch}")

    # Test 2: Different texts in batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 2: Different texts in batch")
    print(f"{'=' * 70}")

    # Single inference for both texts
    single_emb1 = embed_single(tokenizer, model, texts[0], device)
    single_emb2 = embed_single(tokenizer, model, texts[1], device)

    # Batch inference
    batch_embs = embed_batch(tokenizer, model, [texts[0], texts[1]], device)

    diff_1 = torch.abs(single_emb1 - batch_embs[0]).max().item()
    diff_2 = torch.abs(single_emb2 - batch_embs[1]).max().item()
    cos_sim_1 = torch.dot(single_emb1, batch_embs[0]).item()
    cos_sim_2 = torch.dot(single_emb2, batch_embs[1]).item()

    print(f"\n{'Test Case':<40} {'Max Diff':<15} {'Cosine Sim'}")
    print("-" * 70)
    status_1 = "✓ identical" if diff_1 < 0.0001 else "✗ DIFFERENT"
    status_2 = "✓ identical" if diff_2 < 0.0001 else "✗ DIFFERENT"
    print(
        f"{'text[0] single vs batch[0]':<40} {diff_1:<15.6f} {cos_sim_1:.6f} {status_1}"
    )
    print(
        f"{'text[1] single vs batch[1]':<40} {diff_2:<15.6f} {cos_sim_2:.6f} {status_2}"
    )

    # Test 3: Larger batch
    print(f"\n{'=' * 70}")
    print("EXPERIMENT 3: Larger batch (4 texts)")
    print(f"{'=' * 70}")

    single_embs = [embed_single(tokenizer, model, text, device) for text in texts]
    batch_embs = embed_batch(tokenizer, model, texts, device)

    max_diff = max(
        torch.abs(single_embs[i] - batch_embs[i]).max().item() for i in range(4)
    )
    avg_cos_sim = (
        sum(torch.dot(single_embs[i], batch_embs[i]).item() for i in range(4)) / 4
    )

    print(f"Max difference across all 4 texts: {max_diff:.6f}")
    print(f"Average cosine similarity: {avg_cos_sim:.6f}")
    status = "✓ identical" if max_diff < 0.0001 else "✗ DIFFERENT"
    print(f"Result: {status}")

    # Comparison with ONNX
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH ONNX RESULTS")
    print(f"{'=' * 70}")
    print(f"\nONNX quantized results showed:")
    print(f"  - Same text × 2: 0.000000 (identical)")
    print(f"  - Different texts: ~0.5 max diff (significant interference)")
    print(f"\nTransformers results:")
    print(f"  - Same text × 2: {diff_0:.6f} (identical: {diff_0 < 0.0001})")
    print(f"  - Different texts: {diff_1:.6f}, {diff_2:.6f}")
    print(f"  - Cosine similarity: {cos_sim_1:.6f}, {cos_sim_2:.6f}")
    print(
        f"  - Larger batch (4 texts): {max_diff:.6f} (identical: {max_diff < 0.0001})"
    )

    # Conclusion
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")

    all_identical = (
        diff_0 < 0.0001
        and diff_1 < 0.0001
        and diff_2 < 0.0001
        and diff_batch < 0.0001
        and max_diff < 0.0001
    )

    if all_identical:
        print(
            """
✅ Transformers implementation shows NO interference - batching is safe!

This confirms the interference in ONNX is specific to the ONNX export/runtime,
not a fundamental model characteristic. The transformers implementation can
safely batch different texts.
"""
        )
    elif diff_1 < 0.01 and diff_2 < 0.01:
        print(
            """
⚠️  Transformers implementation shows MINIMAL interference (~0.01):
    - Much smaller than ONNX quantized (~0.5)
    - But not perfect like vision model
    - May be acceptable for most use cases
"""
        )
    else:
        print(
            """
❌ Transformers implementation shows SIGNIFICANT interference
    - Similar to ONNX quantized
    - This suggests the model architecture itself has issues
    - Further investigation needed
"""
        )


if __name__ == "__main__":
    main()
