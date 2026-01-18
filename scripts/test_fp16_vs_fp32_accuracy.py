#!/usr/bin/env python3
"""
Test ONNX FP16 vs FP32 accuracy and batch sensitivity.

Compares:
1. Direct embedding differences (FP16 vs FP32)
2. Batch size sensitivity
3. Text length composition sensitivity (long text with short texts)
4. Ordering sensitivity (same texts in different orders)
"""

import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer


def mean_pool(embeddings: np.ndarray, attention_mask: np.ndarray) -> np.ndarray:
    """Mean pooling over non-padding tokens."""
    mask = attention_mask.reshape(-1, 1)
    return (embeddings * mask).sum(axis=0) / mask.sum()


def embed_single(session: ort.InferenceSession, encoding) -> np.ndarray:
    """Embed a single text (batch_size=1)."""
    result = session.run(
        None,
        {
            "input_ids": np.array([encoding.ids], dtype=np.int64),
            "token_type_ids": np.array([encoding.type_ids], dtype=np.int64),
            "attention_mask": np.array([encoding.attention_mask], dtype=np.int64),
        },
    )[0]
    emb = mean_pool(result[0], np.array(encoding.attention_mask))
    # L2 normalize
    norm = np.linalg.norm(emb)
    return emb / norm if norm > 0 else emb


def embed_batch(session: ort.InferenceSession, encodings: list) -> list[np.ndarray]:
    """Embed multiple texts in a single batch."""
    max_len = max(len(e.ids) for e in encodings)

    input_ids, token_type_ids, attention_mask = [], [], []
    for enc in encodings:
        pad_len = max_len - len(enc.ids)
        input_ids.append(list(enc.ids) + [0] * pad_len)
        token_type_ids.append(list(enc.type_ids) + [0] * pad_len)
        attention_mask.append(list(enc.attention_mask) + [0] * pad_len)

    result = session.run(
        None,
        {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "token_type_ids": np.array(token_type_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        },
    )[0]

    embs = [
        mean_pool(result[i], np.array(attention_mask[i])) for i in range(len(encodings))
    ]
    # L2 normalize each
    return [
        emb / np.linalg.norm(emb) if np.linalg.norm(emb) > 0 else emb for emb in embs
    ]


def create_session(model_path: Path):
    """Create ONNX session, handling FP16/Q4F16 optimization requirements."""
    try:
        return ort.InferenceSession(str(model_path))
    except Exception:
        # FP16/Q4F16 may need optimizations disabled
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        )
        return ort.InferenceSession(str(model_path), sess_options)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two embeddings."""
    return np.dot(a, b)


def test_direct_accuracy(fp32_session, fp16_session, tokenizer):
    """Test direct embedding differences between FP32 and FP16."""
    print(f"\n{'=' * 70}")
    print("TEST 1: Direct FP16 vs FP32 Accuracy")
    print(f"{'=' * 70}")

    # Test texts of varying lengths
    texts = [
        "Short",  # ~2 tokens
        "This is a medium length text with several words",  # ~10 tokens
        "This is a much longer text that contains many more words and should provide a good test case for how the model handles different input lengths and compositions in various batch scenarios",  # ~30 tokens
        "A" * 100,  # Very long, repetitive
        "The quick brown fox jumps over the lazy dog. " * 20,  # ~200 tokens
    ]

    encodings = [tokenizer.encode(t) for t in texts]

    print(f"\n{'Text':<50} {'Tokens':<10} {'Max Diff':<15} {'Cosine Sim':<15}")
    print("-" * 90)

    max_diffs = []
    cos_sims = []

    for text, enc in zip(texts, encodings):
        fp32_emb = embed_single(fp32_session, enc)
        fp16_emb = embed_single(fp16_session, enc)

        diff = np.abs(fp32_emb - fp16_emb).max()
        cos_sim = cosine_similarity(fp32_emb, fp16_emb)

        max_diffs.append(diff)
        cos_sims.append(cos_sim)

        text_preview = text[:47] + "..." if len(text) > 50 else text
        print(f"{text_preview:<50} {len(enc.ids):<10} {diff:<15.6f} {cos_sim:<15.6f}")

    print(f"\nSummary:")
    print(f"  Average max diff: {np.mean(max_diffs):.6f}")
    print(f"  Max max diff: {np.max(max_diffs):.6f}")
    print(f"  Average cosine similarity: {np.mean(cos_sims):.6f}")
    print(f"  Min cosine similarity: {np.min(cos_sims):.6f}")

    return max_diffs, cos_sims


def test_batch_size_sensitivity(fp32_session, fp16_session, tokenizer):
    """Test sensitivity to batch size."""
    print(f"\n{'=' * 70}")
    print("TEST 2: Batch Size Sensitivity")
    print(f"{'=' * 70}")

    # Create texts of similar length
    texts = [
        "First text in batch",
        "Second text in batch",
        "Third text in batch",
        "Fourth text in batch",
        "Fifth text in batch",
        "Sixth text in batch",
        "Seventh text in batch",
        "Eighth text in batch",
    ]

    encodings = [tokenizer.encode(t) for t in texts]

    print(f"\n{'Batch Size':<15} {'Max Diff (FP16 vs FP32)':<25} {'Avg Cos Sim':<15}")
    print("-" * 55)

    results = []

    for batch_size in [1, 2, 4, 8]:
        batch_encodings = encodings[:batch_size]

        # FP32
        fp32_embs = embed_batch(fp32_session, batch_encodings)

        # FP16
        fp16_embs = embed_batch(fp16_session, batch_encodings)

        # Compare each embedding
        diffs = [np.abs(fp32_embs[i] - fp16_embs[i]).max() for i in range(batch_size)]
        cos_sims = [
            cosine_similarity(fp32_embs[i], fp16_embs[i]) for i in range(batch_size)
        ]

        max_diff = np.max(diffs)
        avg_cos_sim = np.mean(cos_sims)

        results.append((batch_size, max_diff, avg_cos_sim))
        print(f"{batch_size:<15} {max_diff:<25.6f} {avg_cos_sim:<15.6f}")

    return results


def test_length_composition_sensitivity(fp32_session, fp16_session, tokenizer):
    """Test sensitivity to text length composition in batch."""
    print(f"\n{'=' * 70}")
    print("TEST 3: Text Length Composition Sensitivity")
    print(f"{'=' * 70}")

    # Create texts of very different lengths
    short_text = "Short"
    medium_text = "This is a medium length text"
    long_text = "This is a much longer text that contains many more words and should provide a good test case for how the model handles different input lengths and compositions in various batch scenarios with additional context"
    very_long_text = "The quick brown fox jumps over the lazy dog. " * 50  # ~500 tokens

    texts = {
        "short": short_text,
        "medium": medium_text,
        "long": long_text,
        "very_long": very_long_text,
    }

    encodings = {name: tokenizer.encode(text) for name, text in texts.items()}

    print(f"\n{'Composition':<50} {'Max Diff':<15} {'Avg Cos Sim':<15}")
    print("-" * 80)

    test_cases = [
        ("All short", ["short", "short", "short"]),
        ("All long", ["very_long", "very_long", "very_long"]),
        ("Mixed: 1 long + 2 short", ["very_long", "short", "short"]),
        ("Mixed: 2 long + 1 short", ["very_long", "very_long", "short"]),
        ("Mixed: 1 long + 1 medium + 1 short", ["very_long", "medium", "short"]),
        ("Mixed: 1 short + 1 long + 1 short", ["short", "very_long", "short"]),
        ("Mixed: 1 short + 1 medium + 1 long", ["short", "medium", "very_long"]),
    ]

    results = []

    for name, composition in test_cases:
        batch_encodings = [encodings[c] for c in composition]

        # FP32
        fp32_embs = embed_batch(fp32_session, batch_encodings)

        # FP16
        fp16_embs = embed_batch(fp16_session, batch_encodings)

        # Compare each embedding
        diffs = [
            np.abs(fp32_embs[i] - fp16_embs[i]).max() for i in range(len(composition))
        ]
        cos_sims = [
            cosine_similarity(fp32_embs[i], fp16_embs[i])
            for i in range(len(composition))
        ]

        max_diff = np.max(diffs)
        avg_cos_sim = np.mean(cos_sims)

        results.append((name, max_diff, avg_cos_sim))
        print(f"{name:<50} {max_diff:<15.6f} {avg_cos_sim:<15.6f}")

    return results


def test_ordering_sensitivity(fp32_session, fp16_session, tokenizer):
    """Test sensitivity to text ordering in batch."""
    print(f"\n{'=' * 70}")
    print("TEST 4: Ordering Sensitivity")
    print(f"{'=' * 70}")

    # Create texts of different lengths
    texts = [
        "Short text",
        "This is a medium length text with several words",
        "This is a much longer text that contains many more words and should provide a good test case",
    ]

    encodings = [tokenizer.encode(t) for t in texts]

    print(
        f"\nTesting if same texts in different orders produce different FP16 vs FP32 differences..."
    )

    # Get reference: single inference for each
    fp32_singles = [embed_single(fp32_session, enc) for enc in encodings]
    fp16_singles = [embed_single(fp16_session, enc) for enc in encodings]

    # Reference differences (single inference)
    ref_diffs = [np.abs(fp32_singles[i] - fp16_singles[i]).max() for i in range(3)]
    ref_cos_sims = [
        cosine_similarity(fp32_singles[i], fp16_singles[i]) for i in range(3)
    ]

    print(f"\nReference (single inference):")
    print(f"  Max diff: {np.max(ref_diffs):.6f}")
    print(f"  Avg cosine sim: {np.mean(ref_cos_sims):.6f}")

    # Test different orderings
    print(f"\n{'Ordering':<50} {'Max Diff':<15} {'Avg Cos Sim':<15}")
    print("-" * 80)

    # Test all permutations
    perms = list(permutations([0, 1, 2]))[:6]  # Test first 6 permutations

    results = []

    for perm in perms:
        batch_encodings = [encodings[i] for i in perm]
        order_str = " -> ".join([f"text{perm[i]}" for i in range(3)])

        # FP32
        fp32_embs = embed_batch(fp32_session, batch_encodings)

        # FP16
        fp16_embs = embed_batch(fp16_session, batch_encodings)

        # Compare: each text should match its single inference
        diffs = []
        cos_sims = []
        for i, orig_idx in enumerate(perm):
            diff = np.abs(fp32_embs[i] - fp16_embs[i]).max()
            cos_sim = cosine_similarity(fp32_embs[i], fp16_embs[i])
            diffs.append(diff)
            cos_sims.append(cos_sim)

        max_diff = np.max(diffs)
        avg_cos_sim = np.mean(cos_sims)

        results.append((order_str, max_diff, avg_cos_sim))
        print(f"{order_str:<50} {max_diff:<15.6f} {avg_cos_sim:<15.6f}")

    # Check if ordering affects the differences
    max_diffs = [r[1] for r in results]
    diff_variance = np.var(max_diffs)

    print(f"\nVariance in max diff across orderings: {diff_variance:.10f}")
    if diff_variance < 1e-10:
        print("✓ Ordering does NOT affect FP16 vs FP32 differences")
    else:
        print("⚠️  Ordering may affect FP16 vs FP32 differences")

    return results


def main():
    print("=" * 70)
    print("ONNX FP16 vs FP32 Accuracy and Batch Sensitivity Analysis")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "txt"

    fp32_path = models_dir / "model.onnx"
    fp16_path = models_dir / "model_fp16.onnx"

    if not fp32_path.exists():
        print(f"❌ FP32 model not found: {fp32_path}")
        sys.exit(1)

    if not fp16_path.exists():
        print(f"❌ FP16 model not found: {fp16_path}")
        print(f"   Download with: ./scripts/download_text_models.sh fp16")
        sys.exit(1)

    print("\nLoading models...")
    fp32_session = create_session(fp32_path)
    fp16_session = create_session(fp16_path)
    tokenizer = Tokenizer.from_file("models/txt/tokenizer.json")
    print("✓ Models loaded")

    # Run all tests
    test_direct_accuracy(fp32_session, fp16_session, tokenizer)
    test_batch_size_sensitivity(fp32_session, fp16_session, tokenizer)
    test_length_composition_sensitivity(fp32_session, fp16_session, tokenizer)
    test_ordering_sensitivity(fp32_session, fp16_session, tokenizer)

    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(
        """
Key Questions Answered:

1. How different is FP16 from FP32?
   - Check "Direct FP16 vs FP32 Accuracy" results above
   - Typically: <0.001 max diff, >0.999 cosine similarity

2. How sensitive is FP16 to batch size?
   - Check "Batch Size Sensitivity" results above
   - Should be consistent across batch sizes

3. How sensitive is FP16 to text length composition?
   - Check "Text Length Composition Sensitivity" results above
   - Long texts with short texts should not cause issues

4. How sensitive is FP16 to ordering?
   - Check "Ordering Sensitivity" results above
   - Same texts in different orders should produce same differences
"""
    )


if __name__ == "__main__":
    main()
