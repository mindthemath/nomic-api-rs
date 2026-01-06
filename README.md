# nomic-serve

A fast Rust server for generating text embeddings using the [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) model via ONNX Runtime.

## Quick Start

```bash
# Download model files
make tokenizer.json model_quantized.onnx

# Build the server
make build

# Run (default: NO_BATCH mode on port 8080)
make run

# Test
make health
make test
```

## API

### `GET /health`
Health check endpoint. Returns `OK` with status 200.

### `POST /embed`
Generate embeddings for one or more texts.

**Single text:**
```bash
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"inputs": "Hello world"}'
```

**Multiple texts:**
```bash
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"inputs": ["Hello world", "Goodbye world"]}'
```

**Response:**
```json
{
  "embeddings": [[0.123, 0.456, ...], [0.789, 0.012, ...]],
  "tokens": [4, 4],
  "time_ms": 12.34,
  "batch_mode": "NoBatch"
}
```

- `embeddings`: Array of 768-dimensional vectors
- `tokens`: Token count for each input text
- `time_ms`: Processing time in milliseconds
- `batch_mode`: Current batch processing mode

## Configuration

All configuration is done via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `MODEL` | `model_quantized.onnx` | Path to ONNX model file |
| `TOKENIZER` | `tokenizer.json` | Path to tokenizer file |
| `BATCH_MODE` | `NO_BATCH` | Batch processing mode (see below) |

## Batch Modes

The server supports three batch processing modes, each with different tradeoffs:

### `NO_BATCH` (Default)
```bash
BATCH_MODE=NO_BATCH ./target/release/nomic-serve
```
- Processes each text sequentially in a loop
- **Slowest** for multiple texts
- **Guaranteed correct** results
- Best for: correctness-critical applications, single-text requests

### `SAFE_BATCH`
```bash
BATCH_MODE=SAFE_BATCH ./target/release/nomic-serve
```
- Currently **identical to NO_BATCH** (sequential processing)
- **Exact same results** as NO_BATCH (guaranteed)
- Best for: when you want the option to enable batching later without code changes

> **Note**: SAFE_BATCH was intended to group texts by token count and batch within groups (no padding needed). However, testing revealed that ONNX batched inference produces slightly different results even without padding, due to numerical precision differences in batched matrix operations. For now, SAFE_BATCH uses sequential processing to guarantee identical results.

### `PAD_BATCH`
```bash
BATCH_MODE=PAD_BATCH ./target/release/nomic-serve
```
- Batches all texts together, pads shorter sequences
- **Fastest** for mixed-length batches
- **Slightly different results** (~0.01-0.2 difference in values)
- Best for: high-throughput applications where small differences are acceptable

### Why do batched results differ?

There are two sources of differences in batched inference:

1. **Padding effects**: Shorter sequences get padded to match the longest. While attention masks zero out padding tokens, the attention computation itself is still affected by their presence.

2. **Numerical precision**: Even without padding, batched matrix operations in ONNX/GPU kernels use different accumulation orders than sequential operations, leading to small floating-point differences.

These differences are typically small (~0.01-0.2 in embedding values). For most use cases (similarity search, clustering, classification), they're negligible and cosine similarity between sequential vs batched embeddings remains >0.99.

If you need **exact reproducibility**, use `NO_BATCH` or `SAFE_BATCH`.

## Deployment

### What to ship

The release binary is self-contained. You need to deploy:

```
target/release/nomic-serve   # The compiled binary (~32MB)
model_quantized.onnx         # The ONNX model (~131MB)
tokenizer.json               # The tokenizer (~700KB)
```

Total deployment size: **~164MB**

### Docker Example

```dockerfile
FROM debian:bookworm-slim

# Only standard C/C++ libs needed (libc, libstdc++, libgcc_s, libm)
# These are included in debian:bookworm-slim

WORKDIR /app

COPY target/release/nomic-serve .
COPY model_quantized.onnx .
COPY tokenizer.json .

ENV PORT=8080
ENV BATCH_MODE=SAFE_BATCH

EXPOSE 8080

CMD ["./nomic-serve"]
```

Build and run:
```bash
docker build -t nomic-serve .
docker run -p 8080:8080 nomic-serve
```

### Systemd Service

```ini
# /etc/systemd/system/nomic-serve.service
[Unit]
Description=Nomic Embedding Server
After=network.target

[Service]
Type=simple
User=nomic
WorkingDirectory=/opt/nomic-serve
ExecStart=/opt/nomic-serve/nomic-serve
Environment=PORT=8080
Environment=BATCH_MODE=SAFE_BATCH
Environment=MODEL=/opt/nomic-serve/model_quantized.onnx
Environment=TOKENIZER=/opt/nomic-serve/tokenizer.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

### Performance Tuning

1. **Use SAFE_BATCH** for batch workloads - it provides the best balance of speed and correctness

2. **ONNX optimization** - The server uses `GraphOptimizationLevel::Level3` (maximum optimization)

3. **Session mutex** - The ONNX session is protected by a mutex. For highest throughput with concurrent requests, consider running multiple instances behind a load balancer

4. **Quantized model** - We use the quantized model by default for ~4x smaller size and faster inference with minimal quality loss

## Development

```bash
# Format code
make fmt

# Build release binary
make build

# Run server
make run

# Run tests
make test          # Single text
make test-list     # Multiple texts

# Compare batch modes (run two servers first)
# Terminal 1: PORT=8080 BATCH_MODE=NO_BATCH ./target/release/nomic-serve
# Terminal 2: PORT=8081 BATCH_MODE=SAFE_BATCH ./target/release/nomic-serve
make verify-safebatch
```

## Model Info

- **Model**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Embedding dimension**: 768
- **Max sequence length**: 8192 tokens
- **Pooling**: Mean pooling over non-padding tokens

## License

MIT

