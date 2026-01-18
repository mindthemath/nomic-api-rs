# Multi-stage build for nomic-serve
# Includes text and vision models for multimodal embeddings

# ============================================================================
# Stage 1: Build
# ============================================================================
# Use Ubuntu 22.04 as base to match CUDA runtime GLIBC version
FROM ubuntu:22.04 AS builder

# Install Rust and build dependencies
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN rustup default 1.92.0

WORKDIR /build

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Create a dummy src to build dependencies
RUN mkdir -p src static/swagger-ui && \
    echo "fn main() {}" > src/main.rs && \
    echo "<!-- placeholder -->" > static/swagger-ui/index.html

# Build dependencies (cached layer)
ARG RUST_BUILD_FEATURES=""
RUN cargo build --release ${RUST_BUILD_FEATURES} && rm -rf src static

# Copy source code and static files (needed for include_str! at compile time)
COPY src ./src
COPY static ./static

# Build the actual binary
# Touch source files to ensure cargo sees them as newer than cached artifacts
RUN touch src/main.rs && cargo build --release ${RUST_BUILD_FEATURES}

# Prepare ONNX Runtime libraries for copying to runtime stage
# Copy libraries to a known location so we can reliably copy them later
RUN mkdir -p /build/app/lib && \
    (find /root/.cache/ort.pyke.io -name "libonnxruntime_providers*.so*" -exec cp -L {} /build/app/lib/ \; 2>/dev/null || true) && \
    (find /build/target/release/deps -name "libonnxruntime_providers*.so*" -type l -exec sh -c 'cp -L "$$1" /build/app/lib/ 2>/dev/null || true' _ {} \; || true) && \
    touch /build/app/lib/.keep  # Ensure directory exists even if no libraries found


# ============================================================================
# Stage 2: CPU Runtime
# ============================================================================
FROM debian:bookworm-slim AS runtime-cpu

# Install runtime dependencies (only standard C libraries)
# dumb-init handles signals properly (SIGTERM, SIGINT) for graceful shutdown
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/nomic-serve ./

# Build arguments for model selection
# Default to quantized models for smaller image size
ARG TXT_MODEL_FILE=model_quantized.onnx
ARG IMG_MODEL_FILE=model_quantized.onnx

# Copy text model files (using build arg)
COPY models/txt/${TXT_MODEL_FILE} models/txt/tokenizer.json models/txt/

# Copy vision model files (using build arg)
COPY models/img/${IMG_MODEL_FILE} models/img/

# Default configuration - full multimodal
# CORS: Set CORS_ORIGINS="https://example.com,https://app.example.com" to customize
#       Set DISABLE_CORS=1 to allow all origins
ENV PORT=8080
ENV TOKENIZER=models/txt/tokenizer.json
# Set model paths from build args
# Note: We need to construct the full path here since ENV can reference ARG
ENV TXT_MODEL=models/txt/${TXT_MODEL_FILE}
ENV IMG_MODEL=models/img/${IMG_MODEL_FILE}

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]


# ============================================================================
# Stage 3: GPU Runtime (CUDA)
# ============================================================================
# Use CUDA 12.3.2 with cuDNN 9 to match ONNX Runtime 2.0.0-rc.10 requirements
# CUDA 12.1.0 is deprecated, so we use 12.3.2 which has cuDNN 9 and is not deprecated
FROM nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS runtime-gpu

# Install runtime dependencies
# dumb-init handles signals properly (SIGTERM, SIGINT) for graceful shutdown
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl3 \
    dumb-init \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy binary from builder
COPY --from=builder /build/target/release/nomic-serve ./

# Copy ONNX Runtime CUDA providers libraries (if available)
# These are needed for GPU mode; if not found, server falls back to CPU automatically
# The directory always exists (created in builder stage) so COPY won't fail
COPY --from=builder /build/app/lib/ /app/lib/

# Build arguments for model selection
# Default to quantized models for smaller image size
ARG TXT_MODEL_FILE=model_quantized.onnx
ARG IMG_MODEL_FILE=model_quantized.onnx

# Copy text model files (using build arg)
COPY models/txt/${TXT_MODEL_FILE} models/txt/tokenizer.json models/txt/

# Copy vision model files (using build arg)
COPY models/img/${IMG_MODEL_FILE} models/img/

# GPU mode enabled by default
# Set LD_LIBRARY_PATH to find ONNX Runtime providers
# CORS: Set CORS_ORIGINS="https://example.com,https://app.example.com" to customize
#       Set DISABLE_CORS=1 to allow all origins
ENV PORT=8080
ENV TOKENIZER=models/txt/tokenizer.json
# Set model paths from build args
ENV TXT_MODEL=models/txt/${TXT_MODEL_FILE}
ENV IMG_MODEL=models/img/${IMG_MODEL_FILE}
ENV USE_GPU=1
# Include CUDA libraries (including cuDNN) in LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/app/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}
# Enable ONNX Runtime verbose logging to see which execution provider is used
ENV ORT_LOG_LEVEL=1
ENV ORT_LOG_SEVERITY_LEVEL=1

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]