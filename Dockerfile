# Multi-stage build for nomic-serve
# Supports both CPU and GPU (CUDA) deployments
# Includes text and vision models for multimodal embeddings

# ============================================================================
# Stage 1: Build
# ============================================================================
# Use Ubuntu 22.04 base to match runtime GLIBC version
FROM ubuntu:22.04 AS builder

# Install Rust and build dependencies  
# Use Rust nightly to support edition2024 (required by dependencies)
RUN apt-get update && apt-get install -y \
    curl \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --default-toolchain nightly \
    && rm -rf /var/lib/apt/lists/*
ENV PATH="/root/.cargo/bin:${PATH}"

WORKDIR /build

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Patch Cargo.toml to use ort 2.0.0-rc.2 for CUDA 11.8 compatibility
RUN sed -i 's|version = "2\.0\.0-rc\.10"|version = "2.0.0-rc.2"|' Cargo.toml && \
    sed -i 's|version = "2\.0\.0-rc\.2"|version = "2.0.0-rc.2"|' Cargo.toml && \
    sed -i 's|download-binaries|load-dynamic|' Cargo.toml && \
    echo "✓ Patched Cargo.toml to use ort 2.0.0-rc.2 with load-dynamic"

# Download ONNX Runtime 1.18.1 (CUDA 11.x build for GPU)
# This version supports IR 10 (required by models) and works with CUDA 11.8 drivers
# Compatible with NVIDIA driver 535+ (CUDA 11.8)
# Note: The build without "cuda12" suffix is the CUDA 11.x build
RUN mkdir -p /build/ort-cuda-11-8 && \
    cd /build/ort-cuda-11-8 && \
    curl -L -o onnxruntime-linux-x64-gpu-1.18.1.tgz \
        https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-gpu-1.18.1.tgz && \
    tar -xzf onnxruntime-linux-x64-gpu-1.18.1.tgz

# Download ONNX Runtime 1.18.1 (CPU build for CPU runtime)
RUN mkdir -p /build/ort-cpu && \
    cd /build/ort-cpu && \
    curl -L -o onnxruntime-linux-x64-1.18.1.tgz \
        https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz && \
    tar -xzf onnxruntime-linux-x64-1.18.1.tgz

# Find ONNX Runtime library and set environment variables (for GPU build)
RUN ORT_LIB=$(find /build/ort-cuda-11-8/onnxruntime-linux-x64-gpu-1.18.1/lib -name "libonnxruntime.so*" -type f | head -1) && \
    ORT_LIB_REAL=$(readlink -f "$ORT_LIB" 2>/dev/null || echo "$ORT_LIB") && \
    echo "$ORT_LIB_REAL" > /build/ort-lib-path.txt && \
    echo "$(dirname $ORT_LIB_REAL)" > /build/ort-lib-dir.txt

# Find CPU ONNX Runtime library path
RUN ORT_LIB_CPU=$(find /build/ort-cpu/onnxruntime-linux-x64-1.18.1/lib -name "libonnxruntime.so*" -type f | head -1) && \
    ORT_LIB_CPU_REAL=$(readlink -f "$ORT_LIB_CPU" 2>/dev/null || echo "$ORT_LIB_CPU") && \
    echo "$(dirname $ORT_LIB_CPU_REAL)" > /build/ort-lib-cpu-dir.txt

# Create a dummy src to build dependencies
RUN mkdir -p src static/swagger-ui && \
    echo "fn main() {}" > src/main.rs && \
    echo "<!-- placeholder -->" > static/swagger-ui/index.html

# Build dependencies (cached layer)
# Enable cuda-legacy feature with load-dynamic, using ONNX Runtime 1.18.1 (CUDA 11.x build)
# Note: cuda-legacy feature uses ort rc.2 API (compatible with CUDA 11.8 ONNX Runtime)
# Remove Cargo.lock to force regeneration with rc2
# Update BOTH ort and ort-sys to rc2
# Note: ort rc2 expects 1.17.x but we use 1.18.1 for IR 10 support (version check happens at runtime)
RUN rm -f Cargo.lock && \
    cargo update -p ort --precise 2.0.0-rc.2 && \
    cargo update -p ort-sys --precise 2.0.0-rc.2 && \
    ORT_DYLIB_PATH=$(cat /build/ort-lib-path.txt) && \
    LD_LIBRARY_PATH=$(cat /build/ort-lib-dir.txt):$LD_LIBRARY_PATH && \
    export ORT_DYLIB_PATH LD_LIBRARY_PATH && \
    cargo build --release --features cuda-legacy && rm -rf src static

# Copy source code and static files (needed for include_str! at compile time)
COPY src ./src
COPY static ./static

# Build the actual binary
# Touch source files to ensure cargo sees them as newer than cached artifacts
# Use ONNX Runtime 1.18.1 (CUDA 11.x build) via ORT_DYLIB_PATH
# Note: Using cuda-legacy feature for rc.2 API compatibility (works with CUDA 11.8)
RUN ORT_DYLIB_PATH=$(cat /build/ort-lib-path.txt) && \
    LD_LIBRARY_PATH=$(cat /build/ort-lib-dir.txt):$LD_LIBRARY_PATH && \
    export ORT_DYLIB_PATH LD_LIBRARY_PATH && \
    touch src/main.rs && \
    cargo build --release --features cuda-legacy

# Prepare ONNX Runtime libraries for copying to runtime stages
# GPU libraries (CUDA 11.x build)
RUN mkdir -p /build/app/lib-gpu && \
    find /build/ort-cuda-11-8/onnxruntime-linux-x64-gpu-1.18.1/lib -name "*.so*" -exec cp -L {} /build/app/lib-gpu/ \; 2>/dev/null || true && \
    touch /build/app/lib-gpu/.keep

# CPU libraries
RUN mkdir -p /build/app/lib-cpu && \
    find /build/ort-cpu/onnxruntime-linux-x64-1.18.1/lib -name "*.so*" -exec cp -L {} /build/app/lib-cpu/ \; 2>/dev/null || true && \
    touch /build/app/lib-cpu/.keep

# ============================================================================
# Stage 2: CPU Runtime
# ============================================================================
FROM debian:bookworm-slim AS runtime-cpu

# Build arguments for model selection
# Default to quantized models for smaller image size
ARG TXT_MODEL_FILE=model_quantized.onnx
ARG IMG_MODEL_FILE=model_quantized.onnx

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

# Copy ONNX Runtime CPU libraries
# These are needed for CPU mode execution
COPY --from=builder /build/app/lib-cpu/ /app/lib/

# Copy text model files (using build arg)
COPY models/txt/${TXT_MODEL_FILE} models/txt/tokenizer.json models/txt/

# Copy vision model files (using build arg)
COPY models/img/${IMG_MODEL_FILE} models/img/

# Default configuration - full multimodal
# Set LD_LIBRARY_PATH to find ONNX Runtime libraries
# CORS: Set CORS_ORIGINS="https://example.com,https://app.example.com" to customize
#       Set DISABLE_CORS=1 to allow all origins
ENV PORT=8080
ENV TOKENIZER=models/txt/tokenizer.json
# Set model paths from build args
# Note: We need to construct the full path here since ENV can reference ARG
ENV TXT_MODEL=models/txt/${TXT_MODEL_FILE}
ENV IMG_MODEL=models/img/${IMG_MODEL_FILE}
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]

# ============================================================================
# Stage 3: GPU Runtime (CUDA)
# ============================================================================
# ONNX Runtime CUDA provider for CUDA 11.8
# CUDA 11.8 works with driver 535+ (compatible with older drivers)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04 AS runtime-gpu

# Build arguments for model selection
# Default to quantized models for smaller image size
ARG TXT_MODEL_FILE=model_quantized.onnx
ARG IMG_MODEL_FILE=model_quantized.onnx

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
COPY --from=builder /build/app/lib-gpu/ /app/lib/

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
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]
