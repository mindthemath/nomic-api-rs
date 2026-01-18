# Throughput Optimization Guide

This document provides recommendations for improving requests-per-second (RPS) for the nomic-serve API server.

## Current Architecture Constraints

### Text Model
- **Cannot batch (quantized)**: Severe cross-sample interference (~0.5 diff) proven (see `test_batch_interference.py`)
- **CAN batch (FP32)**: Perfect batching (0.000000 diff) - see `test_text_batch_fp32.py`
- **Sequential processing required for quantized**: Each text must be processed individually
- **137M parameters**: Relatively small model, may be memory-bound on GPU

### Vision Model
- **Batching works (quantized)**: Minor interference (~0.02 diff, 99% similarity) - acceptable
- **Batching perfect (FP32)**: No interference (0.000000 diff, 100% similarity) - verified
- **92M parameters**: Small model, GPU benefits may be limited
- **Fixed input size**: All images preprocessed to 224×224, no padding needed

## GPU Efficiency Analysis

### Why GPU May Not Help

1. **Model Size**: 92M (vision) + 137M (text) = 229M total parameters
   - Small models are often **memory-bound**, not compute-bound
   - GPU overhead (kernel launches, memory transfers) can dominate
   - Your observation: "memory bound - both models took up like half a gig of VRAM"

2. **Batch Size = 1**: 
   - Text model: Must process sequentially (no batching)
   - Vision model: Currently processes sequentially
   - GPU excels with large batches (8+), not single samples

3. **Preprocessing Overhead**:
   - Image decode/resize/crop: CPU-bound
   - Text tokenization: CPU-bound
   - These operations happen before GPU inference

### When GPU Helps

- **Large batches** (8+ images): Parallel processing amortizes overhead
- **Larger models** (500M+ parameters): More compute per sample
- **Mixed workloads**: GPU can process while CPU handles I/O

## Throughput Optimization Strategies

### 1. Horizontal Scaling (Recommended) ⭐

**Best approach for this architecture**: Run multiple server instances behind a load balancer.

#### Setup with nginx

```nginx
# /etc/nginx/sites-available/nomic-serve
upstream nomic_backend {
    least_conn;  # Distribute by connection count
    server localhost:8080;
    server localhost:8081;
    server localhost:8082;
    server localhost:8083;
    server localhost:8084;
    server localhost:8085;
    server localhost:8086;
    server localhost:8087;
}

server {
    listen 80;
    server_name api.example.com;

    location / {
        proxy_pass http://nomic_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
    }
}
```

#### Start Multiple Instances

```bash
# Start 8 instances (one per CPU core)
for port in $(seq 8080 8087); do
    PORT=$port ./target/release/nomic-serve &
done
```

**Expected improvement**: Linear scaling (8 instances ≈ 8x throughput)

**Pros**:
- Simple to implement
- No code changes needed
- Scales horizontally (add more servers)
- Handles failures gracefully (one instance down, others continue)

**Cons**:
- Each instance loads models into memory (8 instances × 0.5GB = 4GB RAM)
- More complex deployment

### 2. Vision Model Batching ✅ (Implemented)

**Verified safe**: `test_vision_batch_interference.py` shows quantized model has minor interference (~0.02 diff, 99% similarity), FP32 is perfect.

**Expected improvement**: 2-3x throughput for vision endpoints (batch_size=8)

**Implementation**: ✅ Already implemented in `/img/batch` endpoint
- Batches multiple images into single ONNX call
- Extracts CLS tokens per sample: `output[batch_idx, 0, :]`
- Handles both quantized (99% similarity) and FP32 (perfect) models

**Note**: Text model can batch with FP32 model (perfect), but quantized model shows severe interference (~0.5 diff).

### 3. Async Request Queuing

**Current**: Tokio handles async I/O, but inference is synchronous.

**Improvement**: Queue requests and process in batches (for vision only):

```rust
// Pseudo-code
struct RequestQueue {
    vision_queue: VecDeque<ImageRequest>,
    batch_size: usize,
    timeout: Duration,
}

// Collect requests until batch_size or timeout
// Process entire batch in single ONNX call
```

**Expected improvement**: 2-3x for vision (if batching works)

**Trade-off**: Adds latency (waiting for batch to fill)

### 4. Preprocessing Optimization

**Current bottlenecks**:
- Image decode (JPEG/PNG decompression)
- Image resize/crop
- Text tokenization

**Optimizations**:
- **Parallel preprocessing**: Use `rayon` for CPU-bound tasks
- **Image decode**: Consider `imageproc` or `opencv` (may be faster than `image` crate)
- **Caching**: Cache preprocessed tensors for repeated images/texts

**Expected improvement**: 10-20% (preprocessing is usually <20% of total time)

### 5. Connection Pooling & Keep-Alive

**Client-side**: Use HTTP keep-alive to reuse connections.

**Server-side**: Already handled by Tokio/Axum (async connection handling).

**Expected improvement**: 5-10% (reduces connection overhead)

### 6. Response Compression

**Add gzip compression** for large responses (embeddings are 768 floats = 3KB):

```rust
use tower_http::compression::CompressionLayer;

let app = Router::new()
    .layer(CompressionLayer::new())
    // ... routes
```

**Expected improvement**: 20-30% for high-latency connections (not localhost)

### 7. Caching

**Cache embeddings** for repeated inputs:

- **In-memory**: `HashMap<text_hash, embedding>` (simple, fast)
- **Redis**: For distributed caching across instances
- **TTL**: Set expiration (e.g., 1 hour)

**Expected improvement**: 10-100x for repeated queries (depends on cache hit rate)

**Use cases**:
- Repeated image URLs
- Common text queries
- Batch requests with duplicates

## Benchmarking

Use the provided scripts to measure improvements:

```bash
# Test vision batching safety
python scripts/test_vision_batch_interference.py

# Benchmark vision batching performance
python scripts/benchmark_vision_batching.py --gpu --batch-sizes 1,4,8,16

# Benchmark API throughput
python scripts/benchmark_throughput.py --concurrent 10 --requests 100
```

## Recommended Architecture

### For High Throughput (100+ RPS)

```
[Load Balancer (nginx)]
    ↓
[8-16 Server Instances] (one per CPU core)
    ↓
[Shared Cache (Redis)] (optional, for repeated queries)
```

**Configuration**:
- 8-16 instances (match CPU cores)
- nginx load balancer (least_conn or round-robin)
- Each instance: `batch_size=1` (text), `batch_size=8` (vision, if verified)
- Optional: Redis cache for repeated queries

### For Low Latency (<50ms P95)

```
[Single Server Instance]
    ↓
[Vision Batching] (batch_size=4-8, timeout=10ms)
    ↓
[GPU] (if batch_size ≥ 4)
```

**Configuration**:
- Single instance (reduces load balancer overhead)
- Vision batching with small batches (4-8) and short timeout
- GPU only if batch_size ≥ 4 (otherwise CPU is faster)

## Cost-Benefit Analysis

| Strategy | Implementation Effort | Expected Gain | Best For |
|----------|----------------------|---------------|----------|
| Horizontal scaling | Low | 8-16x | High throughput |
| Vision batching | ✅ Done | 2-3x | Vision endpoints |
| Text batching (FP32) | Medium | 2-3x | Text endpoints (requires FP32 model) |
| Preprocessing optimization | Medium | 10-20% | All endpoints |
| Caching | Medium | 10-100x* | Repeated queries |
| GPU | High | 0-2x | Large batches only |

*Caching gain depends on cache hit rate

## Conclusion

**For maximum throughput**: Use horizontal scaling (nginx + multiple instances). This is the simplest and most effective approach given the model constraints.

**For vision endpoints**: If batching is verified safe, implement batched inference for additional 2-3x improvement.

**GPU is not recommended** for this use case:
- Models are too small (memory-bound)
- Batch size = 1 for text (no GPU benefit)
- Overhead dominates for small batches

**Next steps**:
1. ✅ Vision batching implemented and verified
2. Consider text batching with FP32 model (if model size acceptable)
3. Set up nginx + multiple instances for horizontal scaling
4. Consider caching for repeated queries

**Model Selection Guide**:
- **Vision**: Quantized model batches acceptably (99% similarity), FP32 perfect
- **Text**: Quantized model cannot batch (use sequential), FP32 batches perfectly
- **Recommendation**: Use FP32 for text if batching needed, quantized for vision is acceptable

