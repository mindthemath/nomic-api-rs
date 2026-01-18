# ONNX FP16 Batching Analysis

## Summary

**ONNX FP16 model batches PERFECTLY** (0.000000 difference, 100% cosine similarity), unlike ONNX INT8 quantization which shows severe interference.

## Test Results

### Text Model: ONNX Variants Comparison

| Variant | Size | Max Diff | Cosine Sim | Verdict |
|---------|------|----------|------------|---------|
| **INT8 (Quantized)** | 131MB | ~0.5-0.6 | ~50-60% | ❌ Unusable |
| **FP16 (Half Precision)** | 262MB | 0.000000 | 100% | ✅ Perfect |
| **FP32 (Full Precision)** | 522MB | 0.000000 | 100% | ✅ Perfect |
| **Q4F16 (4-bit)** | 106MB | 0.000000 | 100% | ✅ Perfect* |

*Q4F16 requires optimizations disabled to load (ONNX Runtime compatibility issue)

## Key Findings

1. **ONNX FP16 batches perfectly**: Shows 0.000000 difference, identical to FP32
2. **INT8 quantization is uniquely problematic**: Only INT8 shows severe interference
3. **Half-precision ≠ quantization**: FP16 (half-precision) is safe, INT8 (quantization) is not
4. **Q4F16 also batches perfectly**: Even 4-bit quantization batches correctly (when it loads)

## Why FP16 Works But INT8 Doesn't

### FP16 (Half Precision)
- **Representation**: 16-bit floating point (1 sign, 5 exponent, 10 mantissa)
- **No quantization**: Direct conversion from FP32, no scales/zero-points
- **Consistent precision**: Same precision across batch
- **Result**: Perfect batching (0.000000 diff)

### INT8 (Quantization)
- **Representation**: 8-bit integer with per-tensor/per-channel scales
- **Dynamic quantization**: Quantization parameters computed from batch statistics
- **Batch-dependent scales**: Different batches may compute different scales
- **Result**: Severe interference (~0.5 diff)

## Technical Details

### ONNX Runtime Compatibility

The FP16 and Q4F16 models require graph optimizations to be disabled:

```python
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession("model_fp16.onnx", sess_options)
```

This is due to an ONNX Runtime compatibility issue with certain graph optimizations in these models. The models still work correctly with optimizations disabled.

## Recommendations

### For Text Model Batching

✅ **Use FP16 model** (recommended):
- Perfect batching (0.000000 diff)
- 2x smaller than FP32 (262MB vs 522MB)
- Good balance of size and performance

✅ **Use FP32 model**:
- Perfect batching (0.000000 diff)
- Largest file size (522MB)
- Maximum precision

❌ **Do NOT use INT8 quantized model**:
- Severe interference (~0.5 diff)
- Unusable for batching
- Use sequential processing (batch_size=1)

⚠️ **Q4F16 model**:
- Perfect batching (0.000000 diff)
- Smallest file size (106MB)
- Requires optimizations disabled (compatibility issue)
- May have slower inference due to disabled optimizations

### Model Selection Guide

| Use Case | Recommended Model | Size | Batching |
|----------|------------------|------|----------|
| **Batching required** | FP16 | 262MB | ✅ Perfect |
| **Maximum precision** | FP32 | 522MB | ✅ Perfect |
| **Sequential only** | INT8 | 131MB | ❌ Cannot batch |
| **Smallest size** | Q4F16 | 106MB | ✅ Perfect* |

*Requires optimizations disabled

## Comparison with PyTorch

| Implementation | Precision | Max Diff | Cosine Sim |
|----------------|-----------|----------|------------|
| **ONNX** | INT8 | ~0.5-0.6 | ~50-60% |
| **ONNX** | FP16 | 0.000000 | 100% |
| **ONNX** | FP32 | 0.000000 | 100% |
| **PyTorch** | FP32 | 0.000000 | 100% |
| **PyTorch** | FP16 | 0.000000-0.000221 | 99.9999-100% |

**Conclusion**: Both ONNX FP16 and PyTorch FP16 batch excellently. The issue is specific to ONNX INT8 quantization.

## Implementation Notes

### Loading FP16 Model

```python
import onnxruntime as ort

# FP16 model requires optimizations disabled
sess_options = ort.SessionOptions()
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
session = ort.InferenceSession("model_fp16.onnx", sess_options)
```

### Rust Implementation

If implementing FP16 support in Rust, you'll need to:

1. Add FP16 model path configuration
2. Disable graph optimizations when loading FP16 model
3. Handle FP16 tensor types (may need conversion to FP32 for some operations)

## Conclusion

**ONNX FP16 is the sweet spot for text model batching**:
- Perfect batching (0.000000 diff)
- 2x smaller than FP32
- No interference issues
- Recommended for production use with batching

The severe interference seen in INT8 quantized models is **not** a general issue with reduced precision - it's specific to INT8 quantization's dynamic parameter computation.

