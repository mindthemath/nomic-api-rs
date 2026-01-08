#!/usr/bin/env python3
"""
Test ONNX FP16 text model for batching interference.

Compares ONNX FP16 vs quantized vs FP32 to see if ONNX FP16 batches better.
"""

import sys
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

    result = session.run(
        None,
        {
            "input_ids": np.array(input_ids, dtype=np.int64),
            "token_type_ids": np.array(token_type_ids, dtype=np.int64),
            "attention_mask": np.array(attention_mask, dtype=np.int64),
        },
    )[0]

    return [
        mean_pool(result[i], np.array(attention_mask[i])) for i in range(len(encodings))
    ]


def test_model(model_path: Path, model_name: str):
    """Test a specific model for interference."""
    print(f"\n{'=' * 70}")
    print(f"Testing: {model_name}")
    print(f"Model: {model_path}")
    print(f"{'=' * 70}")

    if not model_path.exists():
        print(f"❌ Model not found: {model_path}")
        return None

    # Try loading with different options
    try:
        # First try with default settings
        session = ort.InferenceSession(str(model_path))
    except Exception as e:
        print(f"⚠️  Failed to load with default settings: {e}")
        try:
            # Try disabling optimizations
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = (
                ort.GraphOptimizationLevel.ORT_DISABLE_ALL
            )
            session = ort.InferenceSession(str(model_path), sess_options)
            print("✓ Loaded with optimizations disabled")
        except Exception as e2:
            print(f"❌ Failed to load even with optimizations disabled: {e2}")
            return None
    tokenizer = Tokenizer.from_file("models/txt/tokenizer.json")

    # Test texts with varying token counts
    texts = [
        "ONNX in Rust is fast",  # 8 tokens
        "Python is also great",  # 6 tokens
        "Embeddings are useful",  # 8 tokens (same as text 0)
        "Hello world",  # 4 tokens
    ]

    encodings = [tokenizer.encode(t) for t in texts]

    print("\nTest texts:")
    for i, (text, enc) in enumerate(zip(texts, encodings)):
        print(f"  [{i}] {len(enc.ids)} tokens: {text}")

    # Reference: single inference for text 0
    ref_emb = embed_single(session, encodings[0])

    print(f"\n{'Partner':<40} {'Max Diff'}")
    print("-" * 50)

    # Test different batch compositions
    test_cases = [
        ("text[0] × 2 (identical, no padding)", [encodings[0], encodings[0]]),
        ("text[0] + text[2] (same len, NO pad)", [encodings[0], encodings[2]]),
        ("text[0] + text[1] (diff len, padded)", [encodings[0], encodings[1]]),
        ("text[0] + text[3] (diff len, padded)", [encodings[0], encodings[3]]),
        ("text[0] + text[1] + text[2]", [encodings[0], encodings[1], encodings[2]]),
        ("all four texts", encodings),
    ]

    max_diffs = []
    for name, batch_encodings in test_cases:
        batch_embs = embed_batch(session, batch_encodings)
        diff = np.abs(ref_emb - batch_embs[0]).max()
        max_diffs.append(diff)
        status = "✓ identical" if diff < 0.0001 else "✗ DIFFERENT"
        print(f"{name:<40} {diff:.6f} {status}")

    # Calculate cosine similarity for different texts case
    single_emb1 = embed_single(session, encodings[0])
    single_emb2 = embed_single(session, encodings[1])
    batch_embs = embed_batch(session, [encodings[0], encodings[1]])

    diff_1 = np.abs(single_emb1 - batch_embs[0]).max()
    diff_2 = np.abs(single_emb2 - batch_embs[1]).max()

    # Cosine similarity (embeddings are already L2 normalized)
    cos_sim_1 = np.dot(single_emb1, batch_embs[0]) / (
        np.linalg.norm(single_emb1) * np.linalg.norm(batch_embs[0])
    )
    cos_sim_2 = np.dot(single_emb2, batch_embs[1]) / (
        np.linalg.norm(single_emb2) * np.linalg.norm(batch_embs[1])
    )

    return {
        "model_name": model_name,
        "max_diff_same_len": max_diffs[1],  # text[0] + text[2] (same len, no pad)
        "max_diff_different": max_diffs[2],  # text[0] + text[1] (different len)
        "max_diff_1": diff_1,
        "max_diff_2": diff_2,
        "cos_sim_1": cos_sim_1,
        "cos_sim_2": cos_sim_2,
        "all_diffs": max_diffs,
    }


def main():
    print("=" * 70)
    print("Text Model Batching: ONNX FP16 vs Quantized vs FP32")
    print("=" * 70)

    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "txt"

    quantized_path = models_dir / "model_quantized.onnx"
    fp32_path = models_dir / "model.onnx"
    fp16_path = models_dir / "model_fp16.onnx"
    q4f16_path = models_dir / "model_q4f16.onnx"

    results = {}

    # Test quantized model
    if quantized_path.exists():
        results["quantized"] = test_model(quantized_path, "Quantized (INT8)")
    else:
        print(f"\n⚠️  Quantized model not found: {quantized_path}")

    # Test FP32 model
    if fp32_path.exists():
        results["fp32"] = test_model(fp32_path, "FP32 (Full Precision)")
    else:
        print(f"\n⚠️  FP32 model not found: {fp32_path}")

    # Test FP16 model
    if fp16_path.exists():
        results["fp16"] = test_model(fp16_path, "FP16 (Half Precision)")
    else:
        print(f"\n⚠️  FP16 model not found: {fp16_path}")
        print(f"   Download with: ./scripts/download_text_models.sh fp16")

    # Test Q4F16 model (optional, likely worse)
    if q4f16_path.exists():
        results["q4f16"] = test_model(q4f16_path, "Q4F16 (4-bit)")
    else:
        print(f"\n⚠️  Q4F16 model not found: {q4f16_path}")
        print(f"   Download with: ./scripts/download_text_models.sh q4f16")

    # Compare results
    if len(results) >= 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")

        print(f"\n{'Metric':<40} ", end="")
        for name in results.keys():
            print(f"{name:<20}", end="")
        print()
        print("-" * (40 + 20 * len(results)))

        # Max diff same len
        print(f"{'Max diff (same len, no pad)':<40} ", end="")
        for name in results.keys():
            print(f"{results[name]['max_diff_same_len']:<20.6f}", end="")
        print()

        # Max diff different len
        print(f"{'Max diff (different len)':<40} ", end="")
        for name in results.keys():
            print(f"{results[name]['max_diff_different']:<20.6f}", end="")
        print()

        # Cosine similarity
        print(f"{'Cosine sim (text 1)':<40} ", end="")
        for name in results.keys():
            print(f"{results[name]['cos_sim_1']:<20.6f}", end="")
        print()

        print(f"{'Cosine sim (text 2)':<40} ", end="")
        for name in results.keys():
            print(f"{results[name]['cos_sim_2']:<20.6f}", end="")
        print()

        print(f"\n{'=' * 70}")
        print("CONCLUSION")
        print(f"{'=' * 70}")

        if "fp16" in results:
            fp16 = results["fp16"]
            if (
                fp16["max_diff_same_len"] < 0.0001
                and fp16["max_diff_different"] < 0.0001
            ):
                print("\n✅ ONNX FP16 batches PERFECTLY - no interference!")
                print("   ONNX FP16 is safe for batching, unlike INT8 quantized.")
            elif fp16["max_diff_same_len"] < 0.01 and fp16["max_diff_different"] < 0.01:
                print("\n⚠️  ONNX FP16 shows MINIMAL interference (~0.01)")
                print(
                    "   Much better than INT8 quantized (~0.5), acceptable for most use cases."
                )
            elif (
                fp16["max_diff_same_len"]
                < results.get("quantized", {}).get("max_diff_same_len", 1.0) * 0.5
            ):
                print("\n⚠️  ONNX FP16 shows LESS interference than INT8 quantized")
                print("   But still has interference - not perfect like FP32.")
            else:
                print("\n❌ ONNX FP16 shows similar interference to INT8 quantized")
                print("   ONNX FP16 may not be suitable for batching.")

        if "q4f16" in results:
            q4f16 = results["q4f16"]
            quantized = results.get("quantized", {})
            if quantized:
                if q4f16["max_diff_same_len"] > quantized["max_diff_same_len"]:
                    print("\n❌ Q4F16 shows WORSE interference than INT8 quantized")
                    print("   As expected - 4-bit quantization is more aggressive.")
                else:
                    print("\n⚠️  Q4F16 shows similar or better interference than INT8")
                    print("   Unexpected - may need further investigation.")

    elif len(results) == 1:
        model_name = list(results.keys())[0]
        r = results[model_name]
        print(f"\n{'=' * 70}")
        print("RESULTS (Single Model)")
        print(f"{'=' * 70}")
        print(f"\n{model_name}:")
        print(f"  Max diff (same len): {r['max_diff_same_len']:.6f}")
        print(f"  Max diff (different len): {r['max_diff_different']:.6f}")
        print(
            f"  Cosine sim (text 1): {r['cos_sim_1']:.6f} ({r['cos_sim_1']*100:.2f}%)"
        )
        print(
            f"  Cosine sim (text 2): {r['cos_sim_2']:.6f} ({r['cos_sim_2']*100:.2f}%)"
        )


if __name__ == "__main__":
    main()
