# nomic-serve

A fast Rust server for generating text embeddings using the [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) model via ONNX Runtime.

## Quick Start

```bash
# Download model files
make model

# Build
make build

# Run
make run

# Test
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"inputs": "Hello world"}'
```

## API

### `GET /health`
Returns `OK` with status 200.

### `POST /embed`
Generate embeddings for one or more texts.

**Request:**
```json
{"inputs": "Hello world"}
// or
{"inputs": ["Hello world", "Goodbye world"]}
```

**Response:**
```json
{
  "embeddings": [[0.123, 0.456, ...]],
  "tokens": [4],
  "time_ms": 12.34
}
```

- `embeddings`: Array of 768-dimensional vectors (one per input)
- `tokens`: Token count for each input
- `time_ms`: Total processing time in milliseconds

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `MODEL` | `model_quantized.onnx` | Path to ONNX model |
| `TOKENIZER` | `tokenizer.json` | Path to tokenizer |

---

## Why Sequential Processing (No Batching)

This server processes each text **individually** rather than batching multiple texts into a single inference call. This is a deliberate design choice required by the nomic-embed-text-v1.5 ONNX model.

### The Problem: Cross-Sample Interference

We discovered that this model exhibits **cross-sample interference** during batched inference: when multiple texts are processed together, each text's embedding is affected by the other texts in the batch.

**Empirical evidence** (see `debug_batch.py`):

| Batch composition | Max embedding difference from single inference |
|-------------------|-----------------------------------------------|
| Same text × 2 (no padding) | **0.000000** ✓ identical |
| Text A + Text B (same token count, no padding) | **0.539796** ✗ |
| Text A + Text C (different token count, with padding) | **0.570585** ✗ |
| Text A + Text B + Text C | **0.596318** ✗ |

Key findings:
1. **Padding is NOT the cause** — differences occur even when all texts have identical token counts
2. **The same text batched with itself produces identical results** — proving it's cross-sample, not batch-size related
3. **Different texts always interfere** — any batch containing different texts produces different embeddings

### Model Architecture

According to the [Nomic Embed Technical Report](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf):

- **Base architecture**: BERT-based encoder with 137M parameters
- **Context length**: 8,192 tokens (extended from standard 512)
- **Training**: Multi-stage contrastive learning with 235M text pairs
- **Features**: Matryoshka Representation Learning for variable-dimension embeddings

The cross-sample interference is likely caused by one of:

1. **Quantization artifacts**: The ONNX quantized model may compute dynamic quantization parameters across the batch, causing batch-dependent results
2. **ONNX graph optimizations**: Certain fused operations may behave differently for different batch compositions
3. **Normalization layers**: Some normalization computations may inadvertently span the batch dimension

This behavior was verified in Python with `onnxruntime` directly (not just our Rust code), confirming it's a model/runtime characteristic, not an implementation bug.

### Why This Matters

For most embedding use cases (similarity search, clustering, RAG), embeddings need to be **deterministic** — the same text should always produce the same embedding. Cross-sample interference violates this:

```
embed("hello") alone     → [0.123, 0.456, ...]
embed("hello") + "world" → [0.089, 0.512, ...]  # Different!
```

This could cause:
- Inconsistent search results depending on what else was in the batch
- Non-reproducible experiments
- Subtle bugs that are hard to diagnose

### Sequential Processing is Correct

By processing each text individually (batch_size=1), we guarantee:
- **Deterministic results**: Same text → same embedding, always
- **No cross-sample interference**: Each text processed in isolation
- **Correctness over speed**: Throughput is lower, but results are reliable

---

## Scaling for High Throughput

Since batching isn't viable for correctness, here are alternatives for handling high request volumes:

### 1. Horizontal Scaling (Recommended)

Run multiple server instances behind a load balancer:

```bash
# Instance 1
PORT=8080 ./target/release/nomic-serve &

# Instance 2  
PORT=8081 ./target/release/nomic-serve &

# Instance 3
PORT=8082 ./target/release/nomic-serve &
```

Use nginx, HAProxy, or cloud load balancers to distribute requests.

### 2. Process Pool

For CPU-bound workloads, run N instances where N = number of CPU cores:

```bash
for port in $(seq 8080 8087); do
  PORT=$port ./target/release/nomic-serve &
done
```

### 3. Async Request Handling

The server already uses Tokio for async I/O. Multiple concurrent requests are handled efficiently — they just can't share a single inference call.

### 4. Caching

If you have repeated texts, cache embeddings:
- In-memory cache (Redis, memcached)
- Persistent cache (database, vector store)

### 5. Queue-Based Architecture

For high-volume batch jobs:
```
[Requests] → [Queue (Redis/RabbitMQ)] → [Worker Pool] → [Results]
```

Workers process texts sequentially but in parallel across the pool.

---

## Deployment

### Files to Deploy

```
target/release/nomic-serve   # 32MB binary
model_quantized.onnx         # 131MB model
tokenizer.json               # 700KB tokenizer
```

Total: ~164MB

### Dependencies

The binary only requires standard C libraries (glibc, libstdc++). No GPU drivers or CUDA needed for CPU inference.

### Docker

```dockerfile
FROM debian:bookworm-slim
WORKDIR /app
COPY target/release/nomic-serve model_quantized.onnx tokenizer.json ./
ENV PORT=8080
EXPOSE 8080
CMD ["./nomic-serve"]
```

### Systemd

```ini
[Unit]
Description=Nomic Embedding Server
After=network.target

[Service]
Type=simple
WorkingDirectory=/opt/nomic-serve
ExecStart=/opt/nomic-serve/nomic-serve
Environment=PORT=8080
Restart=always

[Install]
WantedBy=multi-user.target
```

---

## Development

```bash
make fmt      # Format code
make build    # Build release binary
make run      # Run server
make test     # Test single embedding
make test-list # Test multiple embeddings
make health   # Health check
```

## Model Info

- **Model**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Embedding dimension**: 768
- **Max sequence length**: 8,192 tokens
- **Pooling**: Mean pooling over non-padding tokens
- **License**: Apache 2.0

## References

- [Nomic Embed Technical Report (2024)](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf)
- [Nomic Embed v1 Blog Post](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
- [HuggingFace Model Card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)

## License

MIT
