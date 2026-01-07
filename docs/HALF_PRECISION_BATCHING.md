# Half-Precision Batching Analysis

## Summary

PyTorch/Transformers implementations using half-precision (FP16/BF16) show **excellent batching behavior**, unlike ONNX INT8 quantization which shows severe interference.

## Test Results

### Text Model: PyTorch Half-Precision

| Precision | Different Texts (2) | Large Batch (4) | Verdict |
|-----------|-------------------|-----------------|---------|
| **FP32** | 0.000000 diff, 100% cos sim | 0.000000 diff, 100% cos sim | Perfect |
| **FP16** | 0.000000 diff, 100% cos sim | 0.000221 diff, 99.9999% cos sim | Excellent |
| **BF16** | 0.000000 diff, 100% cos sim | 0.001940 diff, 99.9938% cos sim | Excellent |

### Comparison with ONNX

| Implementation | Precision | Max Diff | Cosine Sim | Verdict |
|----------------|-----------|----------|------------|---------|
| **ONNX** | INT8 (quantized) | ~0.5-0.6 | ~50-60% | ❌ Unusable |
| **ONNX** | FP32 | 0.000000 | 100% | ✅ Perfect |
| **PyTorch** | FP32 | 0.000000 | 100% | ✅ Perfect |
| **PyTorch** | FP16 | 0.000000-0.000221 | 99.9999-100% | ✅ Excellent |
| **PyTorch** | BF16 | 0.000000-0.001940 | 99.9938-100% | ✅ Excellent |

## Key Findings

1. **PyTorch half-precision is safe**: FP16/BF16 show negligible differences (0.0002-0.002 max diff)
2. **ONNX INT8 is uniquely problematic**: Shows severe interference (~0.5 diff) not seen in PyTorch
3. **Half-precision ≠ quantization**: FP16/BF16 are different from INT8 quantization
4. **Batch size matters slightly**: Larger batches (4+) show tiny differences in FP16/BF16, but still excellent

## Why Half-Precision Works Better

### FP16/BF16 vs INT8 Quantization

**Half-precision (FP16/BF16)**:
- Still uses floating-point representation
- Maintains dynamic range and precision better
- PyTorch handles precision consistently across batch
- Numerical stability preserved

**INT8 Quantization**:
- Integer representation with quantization scales
- Dynamic quantization parameters computed per-batch
- ONNX Runtime optimizations may introduce batch-dependent behavior
- More aggressive precision reduction

### Technical Differences

1. **Representation**:
   - FP16: 16-bit floating point (1 sign, 5 exponent, 10 mantissa)
   - BF16: 16-bit floating point (1 sign, 8 exponent, 7 mantissa)
   - INT8: 8-bit integer with per-tensor/per-channel scales

2. **Quantization**:
   - FP16/BF16: Direct conversion, no quantization parameters
   - INT8: Requires quantization scales and zero-points computed from statistics

3. **Runtime behavior**:
   - PyTorch: Consistent precision handling across batch
   - ONNX Runtime: May optimize differently for different batch compositions

## Recommendations

### For PyTorch/Transformers

✅ **Use FP16/BF16 for batching**:
- Excellent results (99.99%+ similarity)
- Significant memory savings vs FP32
- Faster inference on modern GPUs
- Safe for production use

### For ONNX Runtime

❌ **Avoid INT8 quantization for text model batching**:
- Severe interference (~0.5 diff)
- Use FP32 for batching
- Or use sequential processing (batch_size=1)

✅ **INT8 quantization acceptable for vision model**:
- Minor interference (~0.02 diff, 99% similarity)
- Acceptable for most use cases

## Use Cases

### When to Use Half-Precision

- **GPU inference**: FP16/BF16 are optimized for modern GPUs
- **Memory-constrained environments**: 2x memory savings vs FP32
- **Batch processing**: Excellent batching behavior
- **Production deployments**: Safe and reliable

### When to Use FP32

- **Exact reproducibility**: If you need perfect identical results
- **CPU inference**: FP32 may be faster on CPU
- **Research/experiments**: For maximum precision

### When to Avoid ONNX INT8 (Text Model)

- **Batching required**: Use FP32 or sequential processing
- **Production text embeddings**: Interference too severe
- **Deterministic results needed**: Use FP32

## Implementation Notes

### PyTorch Half-Precision

```python
# FP16
model = model.half()  # Convert to FP16
model = model.to("cuda")

# BF16
model = model.bfloat16()  # Convert to BF16
model = model.to("cuda")
```

### ONNX Quantization

```python
# INT8 quantization (causes interference in text model)
# Use FP32 instead for text model batching
session = ort.InferenceSession("model.onnx")  # FP32
```

## Conclusion

**PyTorch half-precision (FP16/BF16) is safe for batching** and does not exhibit the severe interference seen in ONNX INT8 quantization. The differences are negligible (0.0002-0.002 max diff) and acceptable for production use.

The interference in ONNX INT8 appears to be specific to:
1. ONNX Runtime's quantization implementation
2. Dynamic quantization parameter computation
3. Graph optimizations that interact with quantization

PyTorch's half-precision implementations avoid these issues by maintaining consistent floating-point arithmetic across the batch.

