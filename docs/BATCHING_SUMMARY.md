# Batching Summary: Text vs Vision Models

## Quick Reference

| Model | Quantized ONNX | FP32 ONNX | PyTorch/Transformers |
|-------|---------------|-----------|---------------------|
| **Vision** | ✅ Batches (99% similarity, ~0.02 diff) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) |
| **Text** | ❌ Cannot batch (~50% similarity, ~0.5 diff) | ✅ Perfect (100% similarity) | ✅ Perfect (100% similarity) |

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

**PyTorch/Transformers**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

### Text Model

**Quantized (INT8)**:
- Max difference: ~0.5-0.6
- Cosine similarity: ~50-60%
- **Verdict**: **Unusable for batching** - interference too severe

**FP32 (Full Precision)**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

**PyTorch/Transformers**:
- Max difference: 0.000000
- Cosine similarity: 100%
- **Verdict**: Perfect batching

## Key Insights

1. **Quantization affects text model more than vision**: Text shows ~0.5 diff vs vision's ~0.02 diff
2. **FP32 models batch perfectly**: Both text and vision FP32 models show 0.000000 difference
3. **PyTorch/Transformers batch perfectly**: Confirms interference is ONNX quantization issue, not model architecture
4. **Text model architecture supports batching**: Proven by FP32 and PyTorch implementations

## Recommendations

### For Production

**Vision Model**:
- ✅ Use quantized model with batching (99% similarity acceptable)
- ✅ Use FP32 model for perfect batching (if model size acceptable)

**Text Model**:
- ❌ **DO NOT batch with quantized model** - use sequential processing
- ✅ Use FP32 model for batching (perfect results)
- ✅ Current implementation (sequential) is correct for quantized model

### For Maximum Throughput

1. **Vision endpoints**: Already implemented batching (works with quantized)
2. **Text endpoints**: 
   - Option A: Use FP32 model + implement batching
   - Option B: Keep sequential processing with quantized model
   - Option C: Horizontal scaling (multiple instances)

### Model Size Trade-offs

- **Quantized text**: ~131MB, cannot batch
- **FP32 text**: ~375MB, batches perfectly
- **Quantized vision**: ~93MB, batches acceptably (99%)
- **FP32 vision**: ~375MB, batches perfectly

## Test Scripts

- `test_vision_batch_interference.py` - Vision ONNX batching
- `test_vision_batch_transformers.py` - Vision PyTorch batching
- `test_text_batch_fp32.py` - Text ONNX quantized vs FP32
- `test_text_batch_transformers.py` - Text PyTorch batching
- `compare_quantized_vs_fp32_interference.py` - Vision quantized vs FP32 comparison

## Implementation Status

- ✅ Vision batching implemented in `/img/batch` endpoint
- ❌ Text batching not implemented (quantized model cannot batch)
- 💡 Text batching could be implemented with FP32 model

