# nomic-serve

A fast Rust server for generating text and image embeddings using [nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5) and [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) models via ONNX Runtime.

## Quick Start

```bash
# Download model files
make model-txt model-img

# Build
make build

# Run
make run

# Test
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"input": "Hello world"}'
```

## API

Interactive documentation available at `/docs` (Swagger UI).

### `GET /health`
Returns health status and model availability.

**Response:**
```json
{
  "status": "OK",
  "text_model": true,
  "vision_model": true
}
```

### `GET /info`
Returns server information including model paths and configuration.

**Response:**
```json
{
  "averaging": "geometric",
  "load_cuda": false,
  "txt_model": "models/txt/model_quantized.onnx",
  "tokenizer": "models/txt/tokenizer.json",
  "img_model": "models/img/model_quantized.onnx"
}
```

### `POST /embed` (or `/txt/embed`)
Generate embedding for a single text.

**Request:**
```json
{"input": "Hello world", "dim": 768}
```

- `input` (required): Text to embed
- `dim` (optional): Embedding dimension (1-768). Defaults to 768. Supports [Matryoshka embeddings](https://huggingface.co/blog/matryoshka) - use smaller dimensions for faster similarity search.

**Response:**
```json
{
  "embedding": [0.123, 0.456, ...],
  "tokens": 4,
  "time_ms": 12.34
}
```

**Example with reduced dimension:**
```json
{"input": "Hello world", "dim": 128}
```
Returns a 128-dimensional embedding (faster similarity search, slightly lower quality).

### `POST /batch` (or `/txt/batch`)
Generate embeddings for multiple texts.

**Request:**
```json
{"inputs": ["Hello world", "Goodbye world"], "dim": 8}
```

- `inputs` (required): List of texts to embed
- `dim` (optional): Embedding dimension (1-768). Defaults to 768. Applied to all embeddings in the batch.

**Response:**
```json
{
  "embeddings": [[0.123, 0.456, ...], [0.789, -0.123, ...]],
  "tokens": [4, 5],
  "time_ms": 45.67
}
```

### `POST /img/embed`
Generate embedding for a single image.

**Request:**
```json
{
  "content": "https://example.com/image.jpg",
  "dim": 768
}
```

- `content` (required): Image as URL, data URL (`data:image/jpeg;base64,...`), or raw base64
- `dim` (optional): Embedding dimension (1-768). Defaults to 768.

**Response:**
```json
{
  "embedding": [0.123, 0.456, ...],
  "time_ms": 89.12
}
```

**Limits:** Maximum 20MB per image (compressed). Supports JPEG, PNG, GIF, WebP, BMP, TIFF.

### `POST /img/batch`
Generate embeddings for multiple images.

**Request:**
```json
{
  "contents": [
    "https://example.com/cat.jpg",
    "data:image/png;base64,iVBORw0KGgo..."
  ],
  "dim": 768
}
```

**Response:**
```json
{
  "embeddings": [[0.123, ...], [0.789, ...]],
  "time_ms": 178.45
}
```

### `POST /img/stats`
Extract image statistics including EXIF metadata and color analysis.

**Request:**
```json
{
  "content": "https://example.com/image.jpg",
  "averaging_method": "geometric"
}
```

- `content` (required): Image as URL, data URL (`data:image/jpeg;base64,...`), or raw base64
- `averaging_method` (optional): Color averaging method - `arithmetic` or `geometric` (defaults to `AVERAGING` env var or `geometric`)

**Response:**
```json
{
  "exif_data": {
    "Make": "Canon",
    "Model": "EOS 5D",
    "DateTime": "2024:01:01 12:00:00"
  },
  "color_data": {
    "avg_color": {
      "rgb": [0.5, 0.3, 0.2],
      "hex": "#804d33",
      "method": "geometric"
    },
    "dominant_color": {
      "rgb": [0.6, 0.4, 0.3],
      "hex": "#99664d"
    }
  },
  "time_ms": 12.34
}
```

**Limits:** Maximum 20MB per image (compressed). Supports JPEG, PNG, GIF, WebP, BMP, TIFF.

### `GET /docs`
Swagger UI documentation page.

### `GET /openapi.json`
OpenAPI 3.1.0 schema.

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8080` | Server port |
| `MODEL` | `models/txt/model_quantized.onnx` | Path to text ONNX model (fallback for `TXT_MODEL`) |
| `TXT_MODEL` | `models/txt/model_quantized.onnx` | Path to text ONNX model |
| `TOKENIZER` | `models/txt/tokenizer.json` | Path to tokenizer |
| `IMG_MODEL` | `models/img/model_quantized.onnx` | Path to vision ONNX model |
| `USE_GPU` | `false` | Enable GPU inference (`1` or `true`) |
| `AVERAGING` | `geometric` | Default averaging method for image color statistics: `arithmetic` or `geometric` |
| `DISABLE_CORS` | `false` | Disable CORS entirely (`1` or `true`) |
| `CORS_ORIGINS` | *(see below)* | Comma-separated list of allowed origins |

### CORS Configuration

By default, the server allows requests from localhost only (for local development):
- `http://localhost:3000` / `http://localhost:8080`
- `http://127.0.0.1:3000` / `http://127.0.0.1:8080`

**Production deployment** - set allowed origins explicitly:
```bash
# Allow specific origins (comma-separated, no wildcards)
CORS_ORIGINS="https://example.com,https://app.example.com,https://api.example.com" ./nomic-serve

# Docker
docker run -p 8080:8080 \
  -e CORS_ORIGINS="https://example.com,https://app.example.com" \
  mindthemath/nomic-text-v1.5-rs:latest-cpu
```

**Security notes:**
- Origins must be explicitly listed (no wildcard support)
- If `CORS_ORIGINS` is set but contains invalid values, falls back to localhost defaults (never permissive)
- Invalid origins are silently ignored and logged

**Disable CORS entirely** (allows all origins - use only for internal APIs):
```bash
DISABLE_CORS=1 ./nomic-serve
```

To modify the default allowed origins, edit `DEFAULT_CORS_ORIGINS` in `src/main.rs`.

---

## Matryoshka Embeddings

The nomic-embed-text-v1.5 model supports **Matryoshka embeddings** - variable-dimension embeddings that maintain quality at reduced dimensions. Use the `dim` parameter to truncate embeddings for faster similarity search.

**Benefits:**
- **Faster similarity search**: Smaller vectors = faster distance calculations
- **Reduced storage**: Store fewer dimensions per embedding
- **Quality preserved**: Lower dimensions maintain high quality for most use cases

**Recommended dimensions:**
- `768` (default): Full quality, best for fine-grained tasks
- `512`: ~99% quality, good balance
- `256`: ~95% quality, faster search
- `128`: ~90% quality, very fast search
- `64`: ~85% quality, fastest search

**Example:**
```bash
# Full dimension (default)
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"input": "Hello world"}'

# Reduced dimension for faster search
curl -X POST localhost:8080/embed \
  -H 'content-type: application/json' \
  -d '{"input": "Hello world", "dim": 128}'
```

**Important:** All embeddings in a batch use the same `dim` value. For consistent similarity search, always use the same `dim` for all embeddings you compare.

---

## Why Sequential Processing (No Batching for Quantized Text Model)

This server processes each text **individually** rather than batching multiple texts into a single inference call. This is a deliberate design choice required by the **quantized** nomic-embed-text-v1.5 ONNX model.

### The Problem: Cross-Sample Interference in Quantized Model

We discovered that the **quantized** text model exhibits severe cross-sample interference during batched inference: when multiple texts are processed together, each text's embedding is affected by the other texts in the batch.

**Empirical evidence** (see `test_batch_interference.py` and `test_text_batch_fp32.py`):

| Model | Batch composition | Max embedding difference |
|-------|-------------------|--------------------------|
| **Quantized** | Same text × 2 (no padding) | **0.000000** ✓ identical |
| **Quantized** | Text A + Text B (same token count, no padding) | **0.539796** ✗ severe |
| **Quantized** | Text A + Text C (different token count, with padding) | **0.570585** ✗ severe |
| **FP32** | Any batch composition | **0.000000** ✓ perfect |
| **PyTorch/Transformers** | Any batch composition | **0.000000** ✓ perfect |

**Key findings:**
1. **Quantization is the cause** — FP32 model batches perfectly (0.000000 diff)
2. **Text model is more sensitive** — Quantized text shows ~0.5 diff vs vision's ~0.02 diff
3. **PyTorch/Transformers batches perfectly** — Confirms it's an ONNX quantization issue, not model architecture

### Why Quantization Causes Interference

The quantized (INT8) text model uses dynamic quantization that computes quantization parameters across the batch:
- **Per-batch quantization scales**: Parameters computed from batch statistics
- **Asymmetric quantization**: Zero-point calculations vary with batch composition
- **Dequantization precision**: Rounding errors accumulate differently in batches

The text model's architecture (BERT-based with mean pooling) appears more sensitive to these quantization artifacts than the vision model (ViT with CLS token extraction).

### Model Architecture

According to the [Nomic Embed Technical Report](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf):

- **Base architecture**: BERT-based encoder with 137M parameters
- **Context length**: 8,192 tokens (extended from standard 512)
- **Training**: Multi-stage contrastive learning with 235M text pairs
- **Features**: Matryoshka Representation Learning for variable-dimension embeddings

**Note**: The model architecture itself supports batching (proven by FP32 and PyTorch implementations). The interference is specific to ONNX quantized models.

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

### Sequential Processing is Correct (for Quantized Model)

By processing each text individually (batch_size=1) with the quantized model, we guarantee:
- **Deterministic results**: Same text → same embedding, always
- **No cross-sample interference**: Each text processed in isolation
- **Correctness over speed**: Throughput is lower, but results are reliable

### Alternative: Use FP32 Model for Batching

If you need batching for text embeddings:
- **Use FP32 model** (`model.onnx` instead of `model_quantized.onnx`)
- **FP32 batches perfectly** (0.000000 difference, cosine similarity = 1.0)
- **Trade-off**: Larger model size (~375MB vs ~131MB) but enables batching

**Current implementation**: Uses quantized model by default, so sequential processing is required.

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
target/release/nomic-serve       # 38MB binary (includes CPU + GPU support + image stats)
models/txt/model_quantized.onnx  # 131MB text model
models/txt/tokenizer.json        # 695KB tokenizer
models/img/model_quantized.onnx  # 93MB vision model
```

Total: ~263MB

**Note**: The binary includes both CPU and GPU support. GPU code adds ~2MB but is only loaded when `USE_GPU=1` is set. For CPU-only deployments, you can build without the `cuda` feature to save 2MB (remove `"cuda"` from `Cargo.toml` features).

### Dependencies

**CPU inference**: Standard C libraries (glibc, libstdc++). No GPU drivers or CUDA needed.

**GPU inference**: NVIDIA CUDA drivers and CUDA toolkit. The binary is built with CUDA support enabled by default (can be disabled by removing `"cuda"` feature from `Cargo.toml`).

### Docker

Multi-stage Dockerfile included. Build and push images:

```bash
# Build both CPU and GPU images
make docker-build

# Build specific image
make docker-build-cpu   # CPU-only (debian:bookworm-slim)
make docker-build-gpu   # GPU/CUDA (nvidia/cuda:12.1.0-runtime)

# Push to DockerHub (requires docker login)
make docker-push
```

**Images:**
- `mindthemath/nomic-text-v1.5-rs:latest-cpu` - CPU-only deployment
- `mindthemath/nomic-text-v1.5-rs:latest-gpu` - GPU/CUDA deployment

**Usage:**
```bash
# CPU
docker run -p 8080:8080 --dns 1.1.1.1 --dns 1.0.0.1 mindthemath/nomic-text-v1.5-rs:latest-cpu

# GPU (requires nvidia-docker)
docker run --gpus all -p 8080:8080 --dns 1.1.1.1 --dns 1.0.0.1 mindthemath/nomic-text-v1.5-rs:latest-gpu
```

**Note**: The `--dns` flags are recommended for image embedding endpoints (`/img/embed`, `/img/batch`) to ensure fast DNS resolution. Cloudflare DNS (1.1.1.1) is used for privacy and performance. Without DNS configuration, image URL fetching may be slow (10+ seconds) due to Docker's default DNS configuration.

**Image Size** (as shown by `docker images`):
- **CPU image**: ~358MB
  - Binary: 38MB
  - Text model files (`model_quantized.onnx` + `tokenizer.json`): 132MB
  - Vision model (`model_quantized.onnx`): 93MB
  - Base image (`debian:bookworm-slim`): 74.8MB
  - Runtime dependencies (ca-certificates, libssl3, dumb-init): 9.2MB
  - Layer compression overhead: ~12MB
- **GPU image**: ~2.7GB
  - Binary: 38MB
  - Text model files (`model_quantized.onnx` + `tokenizer.json`): 132MB
  - Vision model (`model_quantized.onnx`): 93MB
  - ONNX Runtime CUDA providers libraries: 196MB
  - CUDA runtime base image (`nvidia/cuda:12.1.0-runtime-ubuntu22.04`): 2.23GB
  - Runtime dependencies: 8MB
  - Layer compression overhead: ~100MB
  - Note: The `-runtime` variant is required (not `-base`) as it includes CUDA runtime libraries needed by ONNX Runtime's CUDA execution provider. The ONNX Runtime CUDA providers add significant size but are required for GPU inference.

**Note**: The CPU Docker image includes both text and vision models for full multimodal support. The GPU image is significantly larger due to the CUDA runtime base image (~2.23GB) required for GPU inference.

**GitHub Actions**: Automatically builds and pushes images on tag releases (e.g., `v1.0.0`).

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
make fmt       # Format code
make build     # Build release binary
make run       # Run server
make test      # Test single embedding
make test-list # Test multiple embeddings
make health    # Health check
```

### Model Variant Comparison

Compare different ONNX model quantizations against the **fp32 baseline** (full precision, unquantized):

```bash
# Download all model variants
make models-all

# Run comparison tests (starts multiple servers, runs tests, cleans up)
make test-models
```

This will:
1. Start servers for each model variant on different ports (8080-8083)
   - Port 8080: `model.onnx` (fp32, **baseline**)
   - Port 8081: `model_quantized.onnx` (quantized)
   - Port 8082: `model_q4f16.onnx` (4-bit quantized)
   - Port 8083: `model_fp16.onnx` (half precision)
2. Run the same test texts through each model
3. Compare embeddings using:
   - **Cosine similarity**: 1.0 = identical, 0.95+ = very similar, <0.9 = different
   - **L2 distance**: Lower is better (0 = identical)
   - **Max/mean absolute differences**: Per-dimension differences
   - **Latency**: Speed comparison

**Interpreting Results:**
- **Cosine similarity > 0.95**: Embeddings are very similar, quantization quality is good
- **Cosine similarity 0.9-0.95**: Moderate differences, may affect fine-grained tasks
- **Cosine similarity < 0.9**: Significant differences, may not be suitable for production
- **Speedup > 1.0**: Faster than baseline (good!)
- **Speedup < 1.0**: Slower than baseline (quantization overhead)

**⚠️ Important: CPU vs GPU Quantization Performance**

Quantization benefits are **primarily on GPU/TPU**, not CPU:

- **On GPU**: Quantized models (INT8, INT4) can be 2-4x faster due to:
  - Specialized tensor cores (e.g., NVIDIA's INT8 cores)
  - Reduced memory bandwidth (smaller model size)
  - Optimized quantized kernels

- **On CPU**: Quantized models are often **slower** because:
  - CPUs lack specialized INT4/INT8 instructions
  - Dequantization overhead negates benefits
  - ONNX Runtime CPU execution provider may not optimize quantized ops
  - Memory bandwidth is rarely the bottleneck on CPU

**Recommendation for CPU inference**: Use **fp32 (`model.onnx`)** for best performance. The quantized models are smaller on disk but don't provide speedups on CPU.

**Note**: If `model_quantized.onnx` shows identical results to fp32 (cosine 1.0), it may be using a lossless quantization scheme or the ONNX Runtime is automatically dequantizing to fp32 for CPU execution.

**Requirements**: Python with `requests` library (`pip install requests` or `pip install -r scripts/requirements.txt`)

The comparison script handles server lifecycle automatically - starts servers, waits for readiness, runs tests, and cleans up on exit (including Ctrl-C).

### GPU Testing

To test model variants on GPU (NVIDIA CUDA):

```bash
# Test all models on GPU
make test-models-gpu
```

**GPU Requirements:**
- NVIDIA GPU with CUDA support
- CUDA drivers installed (`nvidia-smi` should work)
- CUDA toolkit (for building ort crate with CUDA feature - already enabled)
- **ONNX Runtime CUDA providers library**: The ort crate needs `libonnxruntime_providers_shared.so` to be available. If you see "Failed to load library libonnxruntime_providers_shared.so", the server will fall back to CPU automatically.

**Note**: If CUDA libraries aren't found, the server will automatically fall back to CPU execution. Check server logs (`/tmp/nomic-serve-*.log`) to see which execution provider is actually being used.

**Note**: On GPU, quantized models (especially INT8/INT4) should show significant speedups (2-4x) compared to CPU results. The GPU comparison will help you choose the best quantization for your GPU setup.

**GPU Architecture Notes:**
- **RTX 30-series (Ampere)**: INT8 quantization typically performs best (2-3x speedup). fp16 may not show speedups due to conversion overhead or lack of optimized kernels.
- **RTX 40-series / L40S (Ada Lovelace)**: Better fp16/BF16 support with 4th-gen Tensor Cores. fp16 should show better performance than on Ampere.
- **A100/H100**: Excellent fp16/BF16 performance, often matching or exceeding INT8 for many workloads.

**Running server with GPU:**
```bash
# The script automatically sets LD_LIBRARY_PATH, but for manual runs:
ORT_LIB_DIR=$(readlink -f target/release/deps/libonnxruntime_providers_shared.so 2>/dev/null | xargs dirname)
LD_LIBRARY_PATH="$ORT_LIB_DIR:$LD_LIBRARY_PATH" USE_GPU=1 ./target/release/nomic-serve
```

**Note**: The `make test-models-gpu` script automatically finds and sets `LD_LIBRARY_PATH` to point to the ONNX Runtime CUDA providers library. If you run the server manually with `USE_GPU=1`, you may need to set `LD_LIBRARY_PATH` yourself.

## Model Info

**Text Model:**
- **Model**: [nomic-ai/nomic-embed-text-v1.5](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- **Embedding dimension**: 768
- **Max sequence length**: 8,192 tokens
- **Pooling**: Mean pooling over non-padding tokens

**Vision Model:**
- **Model**: [nomic-ai/nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)
- **Embedding dimension**: 768
- **Input size**: 224×224 (auto-resized)
- **Pooling**: CLS token extraction

Both models share the same embedding space via contrastive training, enabling direct comparison of text and image embeddings.

**License**: Apache 2.0

## References

- [Nomic Embed Technical Report (2024)](https://static.nomic.ai/reports/2024_Nomic_Embed_Text_Technical_Report.pdf)
- [Nomic Embed v1 Blog Post](https://www.nomic.ai/blog/posts/nomic-embed-text-v1)
- [HuggingFace Text Model](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5)
- [HuggingFace Vision Model](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5)

## License

MIT
