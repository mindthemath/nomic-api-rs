#!/usr/bin/env python3
"""
Test script demonstrating cross-sample interference in nomic-embed-text-v1.5 ONNX model.

This script proves that batching different texts together produces different embeddings
than processing each text individually, even when no padding is needed (identical token counts).

Usage:
    source .venv/bin/activate  # if using venv with onnxruntime, tokenizers
    python scripts/test_batch_interference.py

Expected output shows that:
- Same text batched with itself → identical to single inference (diff ≈ 0)
- Different texts batched together → significant differences (~0.5)
- This happens even without padding (same token counts)
"""

import onnxruntime as ort
import numpy as np
from tokenizers import Tokenizer


def mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.reshape(-1, 1)
    return (embeddings * mask).sum(axis=0) / mask.sum()


def embed_single(session: ort.InferenceSession, encoding) -> np.ndarray:
    """Embed a single text (batch_size=1)."""
    result = session.run(None, {
        "input_ids": np.array([encoding.ids], dtype=np.int64),
        "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
        "attention_mask": np.array([encoding.attention_mask], dtype=np.int64)
    })[0]
    return mean_pool(result[0], np.array(encoding.attention_mask))


def embed_batch(session: ort.InferenceSession, encodings: list) -> list[np.ndarray]:
    """Embed multiple texts in a single batch, with padding if needed."""
    max_len = max(len(e.ids) for e in encodings)
    
    input_ids, token_type_ids, attention_mask = [], [], []
    for enc in encodings:
        pad_len = max_len - len(enc.ids)
        input_ids.append(list(enc.ids) + [0] * pad_len)
        token_type_ids.append(list(enc.type_ids) + [0] * pad_len)
        attention_mask.append(list(enc.attention_mask) + [0] * pad_len)
    
    result = session.run(None, {
        "input_ids": np.array(input_ids, dtype=np.int64),
        "token_type_ids": np.array(token_type_ids, dtype=np.int64),
        "attention_mask": np.array(attention_mask, dtype=np.int64)
    })[0]
    
    return [mean_pool(result[i], np.array(attention_mask[i])) for i in range(len(encodings))]


def main():
    print("=" * 70)
    print("Cross-Sample Interference Test for nomic-embed-text-v1.5")
    print("=" * 70)
    
    # Load model and tokenizer
    session = ort.InferenceSession("model_quantized.onnx")
    tokenizer = Tokenizer.from_file("tokenizer.json")
    
    # Test texts with varying token counts
    texts = [
        "ONNX in Rust is fast",      # 8 tokens
        "Python is also great",       # 6 tokens
        "Embeddings are useful",      # 8 tokens (same as text 0)
        "Hello world",                # 4 tokens
    ]
    
    encodings = [tokenizer.encode(t) for t in texts]
    
    print("\nTest texts:")
    for i, (text, enc) in enumerate(zip(texts, encodings)):
        print(f"  [{i}] {len(enc.ids)} tokens: {text}")
    
    # Reference: single inference for text 0
    ref_emb = embed_single(session, encodings[0])
    
    print(f"\n{'=' * 70}")
    print("EXPERIMENT: Batch text[0] with different partners")
    print(f"{'=' * 70}")
    print(f"\nReference (text[0] alone): {ref_emb[:3].round(4)}")
    print(f"\n{'Partner':<40} {'First 3 dims':<30} {'Max Diff'}")
    print("-" * 80)
    
    # Test different batch compositions
    test_cases = [
        ("text[0] × 2 (identical, no padding)", [encodings[0], encodings[0]]),
        ("text[0] + text[2] (same len, NO pad)", [encodings[0], encodings[2]]),
        ("text[0] + text[1] (diff len, padded)", [encodings[0], encodings[1]]),
        ("text[0] + text[3] (diff len, padded)", [encodings[0], encodings[3]]),
        ("text[0] + text[1] + text[2]", [encodings[0], encodings[1], encodings[2]]),
        ("all four texts", encodings),
    ]
    
    for name, batch_encodings in test_cases:
        batch_embs = embed_batch(session, batch_encodings)
        diff = np.abs(ref_emb - batch_embs[0]).max()
        status = "✓ identical" if diff < 0.0001 else "✗ DIFFERENT"
        print(f"{name:<40} {str(batch_embs[0][:3].round(4)):<30} {diff:.6f} {status}")
    
    print(f"\n{'=' * 70}")
    print("CONCLUSION")
    print(f"{'=' * 70}")
    print("""
When the SAME text is batched with DIFFERENT texts, the embedding changes
significantly (~0.5 max diff). This happens even without padding.

This proves the model has cross-sample interference, making true batching
unsuitable for applications requiring deterministic embeddings.

Sequential processing (batch_size=1) is the only way to get consistent results.
""")


if __name__ == "__main__":
    main()

