#!/usr/bin/env python3
"""
Test FP32 (unquantized) text model for batching interference.

Compares quantized vs FP32 to see if quantization is the cause of interference.
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
    
    session = ort.InferenceSession(str(model_path))
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
    
    # Cosine similarity
    cos_sim_1 = np.dot(single_emb1, batch_embs[0])
    cos_sim_2 = np.dot(single_emb2, batch_embs[1])
    
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
    print("Text Model Batching: Quantized vs FP32 Comparison")
    print("=" * 70)
    
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models" / "txt"
    
    quantized_path = models_dir / "model_quantized.onnx"
    fp32_path = models_dir / "model.onnx"
    
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
        print(f"   Run: make model-txt (downloads both)")
    
    # Compare results
    if len(results) == 2:
        print(f"\n{'=' * 70}")
        print("COMPARISON")
        print(f"{'=' * 70}")
        
        q = results["quantized"]
        f = results["fp32"]
        
        print(f"\n{'Metric':<40} {'Quantized':<20} {'FP32':<20} {'Difference':<15}")
        print("-" * 95)
        
        print(f"{'Max diff (same len, no pad)':<40} {q['max_diff_same_len']:<20.6f} {f['max_diff_same_len']:<20.6f} {abs(q['max_diff_same_len'] - f['max_diff_same_len']):<15.6f}")
        print(f"{'Max diff (different len)':<40} {q['max_diff_different']:<20.6f} {f['max_diff_different']:<20.6f} {abs(q['max_diff_different'] - f['max_diff_different']):<15.6f}")
        print(f"{'Max diff (image 1)':<40} {q['max_diff_1']:<20.6f} {f['max_diff_1']:<20.6f} {abs(q['max_diff_1'] - f['max_diff_1']):<15.6f}")
        print(f"{'Max diff (image 2)':<40} {q['max_diff_2']:<20.6f} {f['max_diff_2']:<20.6f} {abs(q['max_diff_2'] - f['max_diff_2']):<15.6f}")
        print(f"{'Cosine sim (text 1)':<40} {q['cos_sim_1']:<20.6f} {f['cos_sim_1']:<20.6f} {abs(q['cos_sim_1'] - f['cos_sim_1']):<15.6f}")
        print(f"{'Cosine sim (text 2)':<40} {q['cos_sim_2']:<20.6f} {f['cos_sim_2']:<20.6f} {abs(q['cos_sim_2'] - f['cos_sim_2']):<15.6f}")
        
        print(f"\n{'=' * 70}")
        print("CONCLUSION")
        print(f"{'=' * 70}")
        
        if f['max_diff_same_len'] < 0.0001 and f['max_diff_different'] < 0.0001:
            print("\n✅ FP32 model batches PERFECTLY - no interference!")
            print("   Quantization is the primary cause of interference in text model.")
        elif f['max_diff_same_len'] < q['max_diff_same_len'] * 0.5:
            print("\n⚠️  FP32 model shows LESS interference than quantized")
            print("   But still has interference - not just quantization.")
        else:
            print("\n❌ FP32 model shows similar interference to quantized")
            print("   Interference is NOT primarily due to quantization.")
            print("   Likely caused by ONNX Runtime optimizations.")
    
    elif len(results) == 1:
        model_name = list(results.keys())[0]
        r = results[model_name]
        print(f"\n{'=' * 70}")
        print("RESULTS (Single Model)")
        print(f"{'=' * 70}")
        print(f"\n{model_name}:")
        print(f"  Max diff (same len): {r['max_diff_same_len']:.6f}")
        print(f"  Max diff (different len): {r['max_diff_different']:.6f}")
        print(f"  Cosine sim (text 1): {r['cos_sim_1']:.6f} ({r['cos_sim_1']*100:.2f}%)")
        print(f"  Cosine sim (text 2): {r['cos_sim_2']:.6f} ({r['cos_sim_2']*100:.2f}%)")


if __name__ == "__main__":
    main()

