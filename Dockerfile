# Multi-stage build for nomic-serve
# Includes text and vision models for multimodal embeddings

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
RUN mkdir -p src static/swagger-ui && \
    echo "fn main() {}" > src/main.rs && \
    echo "<!-- placeholder -->" > static/swagger-ui/index.html

# Build dependencies (cached layer)
RUN cargo build --release && rm -rf src static

# Copy source code and static files (needed for include_str! at compile time)
COPY src ./src
COPY static ./static

# Build the actual binary
# Touch source files to ensure cargo sees them as newer than cached artifacts
RUN touch src/main.rs && cargo build --release

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

