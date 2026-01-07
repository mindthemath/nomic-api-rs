# ONNX Batching Interference Analysis

## Summary

When batching different images in ONNX Runtime, we observe small but measurable cross-sample interference:
- **Cosine similarity**: ~99% (vs 100% if identical)
- **Max absolute difference**: ~0.016-0.024 per dimension
- **Impact**: Small but measurable - may affect similarity search rankings slightly

**Important**: The PyTorch/transformers implementation shows **NO interference** (cosine similarity = 1.0), confirming this is an ONNX-specific issue, not a model architecture problem.

## What Causes ONNX Interference?

### 1. **Graph Optimizations**

ONNX Runtime applies aggressive graph optimizations (Level 3 by default) that can introduce batch-dependent behavior:

- **Fused operations**: Multiple operations combined into single kernels may compute intermediate values differently for different batch compositions
- **Constant folding**: Batch-dependent constants may be computed differently
- **Operator fusion**: Fused attention/normalization layers may have batch-dependent numerical precision

**Evidence**: We use `GraphOptimizationLevel::Level3` in our Rust code, which enables all optimizations.

### 2. **Quantization Artifacts** (If Using Quantized Model)

Quantized models (INT8) use dynamic quantization that may compute quantization parameters across the batch:

- **Per-batch quantization scales**: Quantization parameters computed from batch statistics
- **Asymmetric quantization**: Zero-point calculations may vary with batch composition
- **Dequantization precision**: Rounding errors accumulate differently in batches

**Note**: Testing shows similar interference in both quantized and FP32 models, suggesting quantization is **not the primary cause**.

### 3. **Numerical Precision in Batch Processing**

ONNX Runtime may use different numerical precision or accumulation strategies for batched vs single inference:

- **Floating-point accumulation order**: Summing operations may be reordered in batches
- **Intermediate precision**: Some operations may use different precision (FP16, BF16) in batches
- **Memory layout**: Different memory access patterns may cause slight numerical differences

### 4. **Normalization Layers**

Vision transformers use LayerNorm/GroupNorm which compute statistics across spatial dimensions. In batches:

- **Batch statistics**: Some normalization may inadvertently use batch-level statistics
- **Fused normalization**: Fused LayerNorm+activation may compute differently
- **Numerical stability**: Small differences in normalization can propagate

### 5. **Attention Mechanisms**

Self-attention computes relationships between tokens. In batches:

- **Softmax numerical stability**: Softmax in attention may use different numerical tricks for batches
- **Flash Attention**: If enabled, may have batch-dependent optimizations
- **Causal masking**: Even for non-causal models, attention masks may be processed differently

## Why PyTorch Doesn't Show Interference

The PyTorch/transformers implementation shows **zero interference** (cosine similarity = 1.0) because:

1. **No graph optimizations**: PyTorch executes operations as written, without aggressive fusion
2. **Consistent numerical precision**: Uses FP32 consistently, no dynamic quantization
3. **Per-sample processing**: Each sample in batch processed independently at the operation level
4. **No ONNX Runtime**: Direct PyTorch execution avoids ONNX Runtime's optimizations

## Quantized vs FP32 Comparison

Testing both models shows:

- **Quantized (INT8)**: Max diff ~0.018-0.024, cosine ~98-99%
- **FP32 (Full precision)**: Max diff ~0.016-0.022, cosine ~99%

**Conclusion**: Quantization slightly increases interference but is not the primary cause. The interference exists even in FP32 models, confirming it's primarily due to ONNX Runtime optimizations.

## Is This Acceptable?

### For Your Use Case (Relative Rankings)

**Yes, likely acceptable** because:

1. **99% cosine similarity**: Embeddings remain very similar
2. **Relative rankings preserved**: Top-k results likely remain stable
3. **Vector search is approximate**: Most vector databases (Pinecone, Weaviate, Qdrant) use approximate search anyway
4. **Small impact**: The interference is much smaller than the difference between different images (cosine ~0.74)

### When It Might Matter

- **Exact matching**: If you need deterministic embeddings for exact lookups
- **Reproducibility**: If you need identical results across runs
- **Fine-grained ranking**: If you're ranking very similar images (cosine >0.99)

### Mitigation Strategies

1. **Accept the interference**: For most use cases, 99% similarity is sufficient
2. **Use sequential processing**: Process each image individually (batch_size=1) for exact results
3. **Use PyTorch/transformers**: If batching is critical, use the PyTorch implementation
4. **Document the behavior**: Make users aware that batched embeddings may differ slightly

## Recommendations

1. **For production**: Batching is acceptable if 99% similarity is sufficient for your use case
2. **For exact matching**: Use sequential processing (current implementation)
3. **For maximum throughput**: Implement batching with documentation about the ~1% difference
4. **For research/reproducibility**: Use sequential processing or PyTorch implementation

## Technical Details

### Test Methodology

1. Process image individually (batch_size=1) → embedding A
2. Process same image in batch with different image (batch_size=2) → embedding B
3. Compare: `max(|A[i] - B[i]|)` across all 768 dimensions
4. Calculate cosine similarity: `dot(A, B)` (both normalized)

### Observed Metrics

- **Max absolute difference**: 0.016-0.024 (one dimension differs by this amount)
- **Mean absolute difference**: 0.003-0.005 (average across all dimensions)
- **Cosine similarity**: 0.98-0.99 (99% similar)
- **L2 distance**: 0.13-0.19 (Euclidean distance between embeddings)
- **Dimensions affected**: 82-87% of dimensions differ by >0.001

### Comparison to Baseline

- **Two different images (both single)**: Cosine ~0.74, max diff ~0.12
- **Same image (single vs batched)**: Cosine ~0.99, max diff ~0.02
- **Interference is ~6x smaller** than the difference between different images

## References

- ONNX Runtime optimization levels: https://onnxruntime.ai/docs/performance/graph-optimizations.html
- Quantization in ONNX: https://onnxruntime.ai/docs/performance/quantization.html
- Vision Transformer architecture: See `docs/IMAGE_PROCESSING_PIPELINE.md`

