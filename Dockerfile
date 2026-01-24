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





# Ensure models directory exists and is writable for on-demand downloads


RUN mkdir -p /app/models && chmod 777 /app/models





# Default configuration - full multimodal


ENV PORT=8080


ENV TOKENIZER=models/txt/tokenizer.json





EXPOSE 8080





# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)


ENTRYPOINT ["dumb-init", "--"]


CMD ["./nomic-serve"]








# ============================================================================


# Stage 3: GPU Runtime (CUDA)


# ============================================================================


# Use CUDA 12.3.2 with cuDNN 9 to match ONNX Runtime 2.0.0-rc.10 requirements


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


COPY --from=builder /build/app/lib/ /app/lib/





# Ensure models directory exists and is writable for on-demand downloads


RUN mkdir -p /app/models && chmod 777 /app/models





# GPU mode enabled by default


# Set LD_LIBRARY_PATH to find ONNX Runtime providers


ENV PORT=8080


ENV TOKENIZER=models/txt/tokenizer.json


ENV USE_GPU=1


# Include CUDA libraries (including cuDNN) in LD_LIBRARY_PATH


ENV LD_LIBRARY_PATH=/app/lib:/usr/local/cuda/lib64:/usr/local/cuda/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}


# Enable ONNX Runtime verbose logging


ENV ORT_LOG_LEVEL=1


ENV ORT_LOG_SEVERITY_LEVEL=1





EXPOSE 8080





# Use dumb-init to handle signals properly (Ctrl-C, docker stop, etc.)


ENTRYPOINT ["dumb-init", "--"]


CMD ["./nomic-serve"]

