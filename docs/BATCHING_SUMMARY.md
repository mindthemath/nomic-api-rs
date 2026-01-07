# Batching Summary: Text vs Vision Models

## Quick Reference

| Model | ONNX INT8 | ONNX FP16 | ONNX FP32 | PyTorch FP32 | PyTorch FP16/BF16 |
|-------|-----------|-----------|-----------|--------------|-------------------|
| **Vision** | ✅ Batches (99% similarity, ~0.02 diff) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) | ✅ Excellent (99.99%+ similarity) |
| **Text** | ❌ Cannot batch (~50% similarity, ~0.5 diff) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) | ✅ Excellent (99.99%+ similarity) |

## Detailed Findings

### Vision Model

**Quantized (INT8)**:
- Max difference: ~0.018-0.024
- Cosine similarity: ~98-99%
- **Verdict**: Acceptable for most use cases (relative rankings preserved)

**FP32 (Full Precision)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

**PyTorch/Transformers (FP32)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

**PyTorch/Transformers (FP16)**:
- Max difference: 0.000000-0.000221 (depending on batch size)
- Cosine similarity: 99.9999-100%
- **Verdict**: Excellent batching (negligible differences)

**PyTorch/Transformers (BF16)**:
- Max difference: 0.000000-0.001940 (depending on batch size)
- Cosine similarity: 99.9938-100%
- **Verdict**: Excellent batching (very small differences)

### Text Model

**ONNX INT8 (Quantized)**:
- Max difference: ~0.5-0.6
- Cosine similarity: ~50-60%
- **Verdict**: **Unusable for batching** - interference too severe

**ONNX FP16 (Half Precision)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: **Perfect batching** ✅

**ONNX FP32 (Full Precision)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

**PyTorch/Transformers (FP32)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

**PyTorch/Transformers (FP16)**:
- Max difference: 0.000000-0.000221 (depending on batch size)
- Cosine similarity: 99.9999-100%
- **Verdict**: Excellent batching (negligible differences)

**PyTorch/Transformers (BF16)**:
- Max difference: 0.000000-0.001940 (depending on batch size)
- Cosine similarity: 99.9938-100%
- **Verdict**: Excellent batching (very small differences)

## Key Insights

1. **INT8 quantization is uniquely problematic**: Only ONNX INT8 shows severe interference (~0.5 diff) for text model
2. **ONNX FP16 batches perfectly**: ONNX FP16 shows 0.000000 difference, just like FP32
3. **Half-precision ≠ quantization**: FP16 (half-precision) is safe, INT8 (quantization) is not
4. **FP32 models batch perfectly**: Both text and vision FP32 models show 0.000000 difference
5. **PyTorch/Transformers batch perfectly**: Confirms interference is ONNX INT8 quantization issue, not model architecture
6. **Text model architecture supports batching**: Proven by FP32, FP16, and PyTorch implementations
7. **PyTorch half-precision is safe**: PyTorch FP16/BF16 shows excellent batching (99.99%+ similarity)

## Recommendations

### For Production

**Vision Model**:
- ✅ Use quantized model with batching (99% similarity acceptable)
- ✅ Use FP32 model for perfect batching (if model size acceptable)

**Text Model**:
- ❌ **DO NOT batch with INT8 quantized model** - use sequential processing
- ✅ **Use FP16 model for batching** (perfect results, 262MB vs 522MB FP32)
- ✅ Use FP32 model for batching (perfect results, larger file size)
- ✅ Current implementation (sequential) is correct for INT8 quantized model

### For Maximum Throughput

1. **Vision endpoints**: Already implemented batching (works with quantized)
2. **Text endpoints**: 
   - Option A: Use FP32 model + implement batching
   - Option B: Keep sequential processing with quantized model
   - Option C: Horizontal scaling (multiple instances)

### Model Size Trade-offs

- **INT8 quantized text**: ~131MB, cannot batch ❌
- **FP16 text**: ~262MB, batches perfectly ✅ (recommended for batching)
- **FP32 text**: ~522MB, batches perfectly ✅
- **Q4F16 text**: ~106MB, batches perfectly ✅ (but requires optimizations disabled)
- **Quantized vision**: ~93MB, batches acceptably (99%)
- **FP32 vision**: ~375MB, batches perfectly

## Test Scripts

- `test_vision_batch_interference.py` - Vision ONNX batching
- `test_vision_batch_transformers.py` - Vision PyTorch batching
- `test_text_batch_fp32.py` - Text ONNX quantized vs FP32
- `test_text_batch_onnx_fp16.py` - Text ONNX FP16 vs INT8 vs FP32 vs Q4F16
- `test_text_batch_transformers.py` - Text PyTorch FP32 batching
- `test_text_batch_half_precision.py` - Text PyTorch FP16/BF16 batching
- `compare_quantized_vs_fp32_interference.py` - Vision quantized vs FP32 comparison

## Implementation Status

- ✅ Vision batching implemented in `/img/batch` endpoint
- ❌ Text batching not implemented (INT8 quantized model cannot batch)
- 💡 **Text batching could be implemented with FP16 model** (perfect batching, 262MB)
- 💡 Text batching could also use FP32 model (perfect batching, 522MB)

