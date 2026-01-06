# Multi-stage build for nomic-serve
# Supports both CPU and GPU (CUDA) deployments

# ============================================================================
# Stage 1: Build
# ============================================================================
FROM rust:1.92.0-slim AS builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    ca-certificates \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy dependency files
COPY Cargo.toml Cargo.lock ./

# Create a dummy src to build dependencies
RUN mkdir src && echo "fn main() {}" > src/main.rs

# Build dependencies (cached layer)
RUN cargo build --release && rm -rf src

# Copy source code
COPY src ./src

# Build the actual binary
RUN cargo build --release

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

# Copy model files (use quantized by default)
COPY model_quantized.onnx tokenizer.json ./

# Default to CPU mode
ENV PORT=8080
ENV MODEL=model_quantized.onnx
ENV TOKENIZER=tokenizer.json

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]

# ============================================================================
# Stage 3: GPU Runtime (CUDA)
# ============================================================================
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04 AS runtime-gpu

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

# Copy model files
COPY model_quantized.onnx tokenizer.json ./

# GPU mode enabled by default
# Set LD_LIBRARY_PATH to find ONNX Runtime providers
# Note: The CUDA base image provides CUDA runtime libraries
# ONNX Runtime providers will be loaded dynamically if available
# If not found, server automatically falls back to CPU (see logs)
ENV PORT=8080
ENV MODEL=model_quantized.onnx
ENV TOKENIZER=tokenizer.json
ENV USE_GPU=1
ENV LD_LIBRARY_PATH=/app/lib:${LD_LIBRARY_PATH}

EXPOSE 8080

# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)
ENTRYPOINT ["dumb-init", "--"]
CMD ["./nomic-serve"]

