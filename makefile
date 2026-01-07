# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

default: run

.PHONY: model fmt build clean run health docs openapi test test-list test-dim models-all test-models \
        docker-build docker-build-cpu docker-build-gpu docker-push docker-push-cpu docker-push-gpu \
        model-txt model-txt-all model-img model-img-all check-txt check-img check-models \
        test-img test-img-batch test-multimodal test-img-stats run-stats

# ==============================================================================
# Model Files
# ==============================================================================

# Text model (nomic-embed-text-v1.5)
model-txt:
	@bash scripts/download_text_models.sh quantized
	@bash scripts/download_text_models.sh fp32

model-txt-all:
	@bash scripts/download_text_models.sh all

# Vision model (nomic-embed-vision-v1.5)
model-img:
	@bash scripts/download_vision_models.sh quantized
	@bash scripts/download_vision_models.sh fp32

model-img-all:
	@bash scripts/download_vision_models.sh all

# Default: download text model (backward compatibility)
model: model-txt
	@echo "✓ Text model files ready"

# Download all model variants for comparison
models-all: model-txt-all model-img-all
	@echo "✓ All model variants downloaded"

# ==============================================================================
# Validity Checks
# ==============================================================================

check-txt:
	@if [ ! -f "models/txt/model_quantized.onnx" ]; then \
		echo "❌ Text model not found. Run: make model-txt"; \
		exit 1; \
	fi
	@if [ ! -f "models/txt/tokenizer.json" ]; then \
		echo "❌ Tokenizer not found. Run: make model-txt"; \
		exit 1; \
	fi
	@echo "✓ Text model files present"

check-img:
	@if [ ! -f "models/img/model_quantized.onnx" ]; then \
		echo "❌ Vision model not found. Run: make model-img"; \
		exit 1; \
	fi
	@echo "✓ Vision model files present"

check-models: check-txt check-img
	@echo "✓ All model files present"

# ==============================================================================
# Build
# ==============================================================================

fmt:
	cargo fmt

target/release/nomic-serve: src/main.rs Cargo.toml static/swagger-ui/index.html
	cargo build --release --features ort-rc2-api

build: fmt
	@# Ensure ort is rc.2 (compatible with ONNX Runtime 1.18.1)
	@# Use ort-rc2-api feature to get rc.2 API style (no CUDA code compiled)
	@cargo update -p ort --precise 2.0.0-rc.2
	@cargo update -p ort-sys --precise 2.0.0-rc.2
	cargo build --release --features ort-rc2-api
	@echo "✓ Build complete (CPU only, using ort rc.2 API)"

# Build with CUDA support (Linux only, requires NVIDIA drivers)
build-cuda: fmt
	cargo build --release --features cuda
	@echo "✓ Build complete (with CUDA support)"

# Build with CUDA 12.2 support - see build-cuda-legacy target below (after ONNX Runtime setup)

check:
	@cargo check
	@echo "✓ Check complete"

lint:
	@uvx black scripts/
	@uvx isort --profile black scripts/
	@echo "✓ Lint complete"

clean:
	rm -rf target Cargo.lock

clean-imgs:
	@printf "Are you sure you want to delete scripts/test_images? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]*) rm -rf scripts/test_images; echo "✓ Deleted scripts/test_images"; ;; \
		*) echo "Cancelled."; ;; \
	esac

clean-results:
	@printf "Are you sure you want to delete scripts/results? [y/N] "; \
	read REPLY; \
	case "$$REPLY" in \
		[Yy]*) rm -rf scripts/results; echo "✓ Deleted scripts/results"; ;; \
		*) echo "Cancelled."; ;; \
	esac

# ==============================================================================
# ONNX Runtime for CPU and CUDA
# ==============================================================================

# Directory for downloaded ONNX Runtime
ORT_CUDA_12_2_DIR := $(shell pwd)/.ort-cuda-legacy
ORT_CPU_DIR := $(shell pwd)/.ort-cpu

# Use Python's onnxruntime-gpu libraries (if installed)
# Note: Python package embeds the main library, so we use the downloaded Microsoft release instead
setup-ort-python:
	@echo "⚠️  Python onnxruntime embeds the main library in the Python extension."
	@echo "   Using downloaded Microsoft release instead (recommended)."
	@echo "   Run: make download-ort-cuda-legacy"
	@exit 1

# Download Microsoft's ONNX Runtime 1.17.3 (supports CUDA 11.8+, compatible with ort 2.0.0-rc.2)
# Note: For local builds, 1.17.3 works. Docker uses 1.18.1 for IR 10 support.
# Release: https://github.com/microsoft/onnxruntime/releases/tag/v1.17.3
download-ort-cuda-legacy:
	@echo "Downloading ONNX Runtime 1.17.3 for CUDA 11.8+ (compatible with ort 2.0.0-rc.2)..."
	@mkdir -p $(ORT_CUDA_12_2_DIR)
	@cd $(ORT_CUDA_12_2_DIR) && \
	if [ ! -f "onnxruntime-linux-x64-gpu-1.17.3.tgz" ]; then \
		echo "Downloading from GitHub releases..."; \
		curl -L -o onnxruntime-linux-x64-gpu-1.17.3.tgz \
			https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-gpu-1.17.3.tgz || \
		(echo "❌ Download failed. Trying alternative URL..." && \
		 curl -L -o onnxruntime-linux-x64-gpu-1.17.3.tgz \
			https://github.com/microsoft/onnxruntime/releases/download/v1.17.3/onnxruntime-linux-x64-1.17.3.tgz); \
	fi
	@cd $(ORT_CUDA_12_2_DIR) && \
	if [ ! -d "onnxruntime-linux-x64-gpu-1.17.3" ]; then \
		echo "Extracting..."; \
		tar -xzf onnxruntime-linux-x64-gpu-1.17.3.tgz || \
		(echo "❌ Extraction failed. File may be corrupted. Removing..." && \
		 rm -f onnxruntime-linux-x64-gpu-1.17.3.tgz && exit 1); \
	fi
	@ORT_LIB=$$(find $(ORT_CUDA_12_2_DIR)/onnxruntime-linux-x64-gpu-1.17.3/lib -name "libonnxruntime.so*" -type f | grep -E "(libonnxruntime\.so\.|libonnxruntime\.so$$)" | head -1); \
	if [ -z "$$ORT_LIB" ]; then \
		echo "❌ Could not find libonnxruntime.so in extracted archive"; \
		echo "   Contents of lib/:"; \
		ls -la $(ORT_CUDA_12_2_DIR)/onnxruntime-linux-x64-gpu-1.17.3/lib/ 2>/dev/null || true; \
		exit 1; \
	fi; \
	# Resolve symlink to actual file \
	ORT_LIB_REAL=$$(readlink -f "$$ORT_LIB" 2>/dev/null || echo "$$ORT_LIB"); \
	echo "✓ Found ONNX Runtime library at: $$ORT_LIB_REAL"; \
	echo "export ORT_DYLIB_PATH=$$ORT_LIB_REAL" > .ort-env.sh; \
	echo "export LD_LIBRARY_PATH=$$(dirname $$ORT_LIB_REAL):\$$LD_LIBRARY_PATH" >> .ort-env.sh; \
	echo "✓ Created .ort-env.sh with ORT_DYLIB_PATH and LD_LIBRARY_PATH"; \
	echo "   Source it with: source .ort-env.sh"

# Auto-setup: download Microsoft release (recommended)
setup-ort-cuda-legacy: download-ort-cuda-legacy

# Build with CUDA 11.8+ support (uses ort 2.0.0-rc.2 + ONNX Runtime 1.17.3)
build-cuda-legacy: fmt
	@# Patch Cargo.toml to use ort 2.0.0-rc.2 for CUDA 11.8+ compatibility
	@sed -i.bak 's|version = "2\.0\.0-rc\.10"|version = "2.0.0-rc.2"|' Cargo.toml
	@if [ -f "$(shell pwd)/.ort-env.sh" ]; then \
		echo "Loading ORT_DYLIB_PATH from .ort-env.sh..."; \
		export $$(grep -v '^#' $(shell pwd)/.ort-env.sh | xargs); \
	fi; \
	if [ -z "$$ORT_DYLIB_PATH" ]; then \
		echo "❌ ORT_DYLIB_PATH not set. Run: make download-ort-cuda-legacy"; \
		mv Cargo.toml.bak Cargo.toml 2>/dev/null || true; \
		exit 1; \
	fi; \
	if [ ! -f "$$ORT_DYLIB_PATH" ]; then \
		echo "❌ ORT_DYLIB_PATH points to non-existent file: $$ORT_DYLIB_PATH"; \
		echo "   Run: make download-ort-cuda-legacy"; \
		mv Cargo.toml.bak Cargo.toml 2>/dev/null || true; \
		exit 1; \
	fi; \
	echo "✓ Using ONNX Runtime 1.17.3 from: $$ORT_DYLIB_PATH"; \
	rm -f Cargo.lock; \
	ORT_DYLIB_PATH=$$ORT_DYLIB_PATH LD_LIBRARY_PATH=$$LD_LIBRARY_PATH cargo build --release --features cuda-legacy || (mv Cargo.toml.bak Cargo.toml 2>/dev/null || true; exit 1); \
	mv Cargo.toml.bak Cargo.toml 2>/dev/null || true
	@echo "✓ Build complete (with CUDA 11.8+ support, using ort 2.0.0-rc.2 + ONNX Runtime 1.17.3)"

# Run with CUDA 11.8+ (auto-loads .ort-env.sh)
run-gpu: build-cuda-legacy check-models
	@if [ -f "$(shell pwd)/.ort-env.sh" ]; then \
		export $$(grep -v '^#' $(shell pwd)/.ort-env.sh | xargs); \
	fi; \
	USE_GPU=1 ORT_DYLIB_PATH=$$ORT_DYLIB_PATH LD_LIBRARY_PATH=$$LD_LIBRARY_PATH ./target/release/nomic-serve

# Download CPU ONNX Runtime 1.18.1 (for local CPU builds)
download-ort-cpu:
	@echo "Downloading ONNX Runtime 1.18.1 for CPU (supports IR 10)..."
	@mkdir -p $(ORT_CPU_DIR)
	@cd $(ORT_CPU_DIR) && \
	if [ ! -f "onnxruntime-linux-x64-1.18.1.tgz" ]; then \
		echo "Downloading from GitHub releases..."; \
		curl -L -o onnxruntime-linux-x64-1.18.1.tgz \
			https://github.com/microsoft/onnxruntime/releases/download/v1.18.1/onnxruntime-linux-x64-1.18.1.tgz; \
	fi
	@cd $(ORT_CPU_DIR) && \
	if [ ! -d "onnxruntime-linux-x64-1.18.1" ]; then \
		echo "Extracting..."; \
		tar -xzf onnxruntime-linux-x64-1.18.1.tgz || \
		(echo "❌ Extraction failed. File may be corrupted. Removing..." && \
		 rm -f onnxruntime-linux-x64-1.18.1.tgz && exit 1); \
	fi
	@ORT_LIB=$$(find $(ORT_CPU_DIR)/onnxruntime-linux-x64-1.18.1/lib -name "libonnxruntime.so*" -type f | grep -E "(libonnxruntime\.so\.|libonnxruntime\.so$$)" | head -1); \
	if [ -z "$$ORT_LIB" ]; then \
		echo "❌ Could not find libonnxruntime.so in extracted archive"; \
		echo "   Contents of lib/:"; \
		ls -la $(ORT_CPU_DIR)/onnxruntime-linux-x64-1.18.1/lib/ 2>/dev/null || true; \
		exit 1; \
	fi; \
	ORT_LIB_REAL=$$(readlink -f "$$ORT_LIB" 2>/dev/null || echo "$$ORT_LIB"); \
	echo "✓ Found ONNX Runtime library at: $$ORT_LIB_REAL"; \
	echo "export ORT_DYLIB_PATH=$$ORT_LIB_REAL" > .ort-env-cpu.sh; \
	echo "export LD_LIBRARY_PATH=$$(dirname $$ORT_LIB_REAL):\$$LD_LIBRARY_PATH" >> .ort-env-cpu.sh; \
	echo "✓ Created .ort-env-cpu.sh with ORT_DYLIB_PATH and LD_LIBRARY_PATH"; \
	echo "   Source it with: source .ort-env-cpu.sh"

# Clean downloaded ONNX Runtime
clean-ort:
	rm -rf $(ORT_CUDA_12_2_DIR) $(ORT_CPU_DIR) .ort-env.sh .ort-env-cpu.sh

# ==============================================================================
# Run
# ==============================================================================

run: build check-models
	@# Try to load ORT_DYLIB_PATH from .ort-env-cpu.sh if available
	@if [ -f "$(shell pwd)/.ort-env-cpu.sh" ]; then \
		echo "Loading ORT_DYLIB_PATH from .ort-env-cpu.sh..."; \
		export $$(grep -v '^#' $(shell pwd)/.ort-env-cpu.sh | xargs); \
	fi; \
	if [ -z "$$ORT_DYLIB_PATH" ]; then \
		echo "⚠️  ORT_DYLIB_PATH not set. Trying to use system ONNX Runtime..."; \
		echo "   If this fails, run: make download-ort-cpu"; \
	fi; \
	ORT_DYLIB_PATH=$$ORT_DYLIB_PATH LD_LIBRARY_PATH=$$LD_LIBRARY_PATH ./target/release/nomic-serve

run-full: build check-models
	AVERAGING=arithmetic TXT_MODEL=models/txt/model.onnx IMG_MODEL=models/img/model.onnx ./target/release/nomic-serve

# run-gpu target is defined below after ONNX Runtime setup targets

# Run server (image-stats is now always included, no model files required for /img/stats)
run-stats: build
	./target/release/nomic-serve

# ==============================================================================
# Test - Text Endpoints
# ==============================================================================

health:
	@curl -s http://localhost:8080/health | jq .

docs:
	@echo "Opening docs at http://localhost:8080/docs"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8080/docs

openapi:
	@curl -s http://localhost:8080/openapi.json | jq '.info'

test:
	@echo "Testing /txt/embed..."
	@curl -s -X POST localhost:8080/txt/embed \
		-H 'content-type: application/json' \
		-d '{"input": "ONNX in Rust is fast"}' | \
		jq '{tokens: .tokens, time_ms: (.time_ms | floor), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

test-list:
	@echo "Testing /txt/batch..."
	@curl -s -X POST localhost:8080/txt/batch \
		-H 'content-type: application/json' \
		-d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
		jq '{count: (.embeddings | length), tokens, time_ms: (.time_ms | floor), samples: [.embeddings[] | .[0:3] | map(. * 1000 | floor / 1000)]}'

test-dim:
	@echo "Testing Matryoshka embeddings (dim=128)..."
	@curl -s -X POST localhost:8080/txt/embed \
		-H 'content-type: application/json' \
		-d '{"input": "ONNX in Rust is fast", "dim": 128}' | \
		jq '{tokens: .tokens, time_ms: (.time_ms | floor), dim: (.embedding | length), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

test-query:
	@echo "Testing /txt/query (enforced search_query prefix)..."
	@curl -s -X POST localhost:8080/txt/query \
		-H 'content-type: application/json' \
		-d '{"input": "What is machine learning?"}' | \
		jq '{tokens: .tokens, time_ms: (.time_ms | floor), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

# ==============================================================================
# Test - Image Endpoints
# ==============================================================================

test-img:
	@echo "Testing /img/embed with URL..."
	@curl -s -X POST localhost:8080/img/embed \
		-H 'content-type: application/json' \
		-d '{"content": "https://picsum.photos/400/300"}' | \
		jq '{time_ms: (.time_ms | floor), dim: (.embedding | length), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

test-img-batch:
	@echo "Testing /img/batch with multiple URLs..."
	@curl -s -X POST localhost:8080/img/batch \
		-H 'content-type: application/json' \
		-d '{"contents": ["https://picsum.photos/400/300", "https://picsum.photos/300/400"]}' | \
		jq '{count: (.embeddings | length), time_ms: (.time_ms | floor), samples: [.embeddings[] | .[0:3] | map(. * 1000 | floor / 1000)]}'

# Test image stats endpoint (requires image-stats feature)
test-img-stats:
	@echo "Testing /img/stats with URL (geometric mean)..."
	@curl -s -X POST localhost:8080/img/stats \
		-H 'content-type: application/json' \
		-d '{"content": "https://picsum.photos/400/300", "averaging_method": "geometric"}' | \
		jq '{time_ms: (.time_ms | floor), exif_fields: (.exif_data | keys | length), avg_color: .color_data.avg_color, dominant_color: .color_data.dominant_color}'

test-img-stats-arithmetic:
	@echo "Testing /img/stats with arithmetic mean..."
	@curl -s -X POST localhost:8080/img/stats \
		-H 'content-type: application/json' \
		-d '{"content": "https://picsum.photos/400/300", "averaging_method": "arithmetic"}' | \
		jq '{time_ms: (.time_ms | floor), avg_color: .color_data.avg_color, dominant_color: .color_data.dominant_color}'

# Validate Rust image-stats against Python reference
test-img-stats-validate:
	@echo "Validating Rust /img/stats against Python reference..."
	@cd scripts && time python3 test_image_stats.py --rust-url http://localhost:8080 --count 100 --seed 1231 --tidy --paged

# ==============================================================================
# Test - Multimodal
# ==============================================================================

test-multimodal:
	@echo "Testing multimodal similarity (text vs image)..."
	@echo "Embedding text: 'a photo of a landscape'..."
	@TXT=$$(curl -s -X POST localhost:8080/txt/embed \
		-H 'content-type: application/json' \
		-d '{"input": "search_query: a photo of a landscape"}' | jq -c '.embedding') && \
	echo "Embedding image: random landscape..." && \
	IMG=$$(curl -s -X POST localhost:8080/img/embed \
		-H 'content-type: application/json' \
		-d '{"content": "https://picsum.photos/400/300"}' | jq -c '.embedding') && \
	echo "Computing cosine similarity..." && \
	python3 -c "import json; t=json.loads('$$TXT'); i=json.loads('$$IMG'); dot=sum(a*b for a,b in zip(t,i)); print(f'Cosine similarity: {dot:.4f}')"

# ==============================================================================
# Model Comparison
# ==============================================================================

# Compare all model variants against baseline (model.onnx, fp32)
# Requires: models-all, build, and Python requests library
test-models: build model-txt-all
	@echo "Starting model variant comparison (CPU)..."
	@USE_GPU=0 bash scripts/run_model_comparison.sh

# Compare all model variants on GPU
# Requires: models-all, build, CUDA drivers, and Python requests library
test-models-gpu: build model-txt-all
	@echo "Starting model variant comparison (GPU)..."
	@USE_GPU=1 bash scripts/run_model_comparison.sh

# ==============================================================================
# Docker
# ==============================================================================

DOCKER_IMAGE = mindthemath/nomic-embed-v1.5-rs
DOCKER_TAG ?= latest

# Build both CPU and GPU images
docker-build: docker-build-cpu docker-build-gpu

# Build CPU-only image (requires both models, defaults to quantized)
docker-build-cpu: model-txt model-img
	@echo "Building CPU Docker image (quantized models)..."
	docker build --target runtime-cpu \
		--build-arg TXT_MODEL_FILE=model_quantized.onnx \
		--build-arg IMG_MODEL_FILE=model_quantized.onnx \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu -t $(DOCKER_IMAGE):latest-cpu .

# Build CPU image with full precision models
docker-build-cpu-full: model-txt model-img
	@echo "Building CPU Docker image (full precision models)..."
	docker build --target runtime-cpu \
		--build-arg TXT_MODEL_FILE=model.onnx \
		--build-arg IMG_MODEL_FILE=model.onnx \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu-full -t $(DOCKER_IMAGE):latest-cpu-full .

docker-run-cpu: docker-build-cpu
	docker run -p 8080:8080 --dns 1.1.1.1 --dns 1.0.0.1 $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu

# Build GPU (CUDA) image (requires both models, defaults to quantized)
docker-build-gpu: model-txt model-img
	@echo "Building GPU Docker image (quantized models)..."
	docker build --target runtime-gpu \
		--build-arg TXT_MODEL_FILE=model_quantized.onnx \
		--build-arg IMG_MODEL_FILE=model_quantized.onnx \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu -t $(DOCKER_IMAGE):latest-gpu .

# Build GPU image with full precision models
docker-build-gpu-full: model-txt model-img
	@echo "Building GPU Docker image (full precision models)..."
	docker build --target runtime-gpu \
		--build-arg TXT_MODEL_FILE=model.onnx \
		--build-arg IMG_MODEL_FILE=model.onnx \
		-t $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu-full -t $(DOCKER_IMAGE):latest-gpu-full .

docker-run-gpu: docker-build-gpu
	docker run --gpus all -p 8080:8080 --dns 1.1.1.1 --dns 1.0.0.1 $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu

# Push both images
docker-push: docker-push-cpu docker-push-gpu

# Push CPU image
docker-push-cpu: docker-build-cpu
	@echo "Pushing CPU image to DockerHub..."
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu
	docker push $(DOCKER_IMAGE):latest-cpu

# Push GPU image
docker-push-gpu: docker-build-gpu
	@echo "Pushing GPU image to DockerHub..."
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu
	docker push $(DOCKER_IMAGE):latest-gpu
