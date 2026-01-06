# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

.PHONY: model fmt build clean run health docs openapi test test-list test-dim models-all test-models \
        docker-build docker-build-cpu docker-build-gpu docker-push docker-push-cpu docker-push-gpu \
        model-txt model-txt-all model-img model-img-all check-txt check-img check-models \
        test-img test-img-batch test-multimodal

# ==============================================================================
# Model Files
# ==============================================================================

# Text model (nomic-embed-text-v1.5)
model-txt:
	@bash scripts/download_text_models.sh defualt fp32

model-txt-all:
	@bash scripts/download_text_models.sh all

# Vision model (nomic-embed-vision-v1.5)
model-img:
	@bash scripts/download_vision_models.sh default fp32

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
	cargo build --release

build: fmt target/release/nomic-serve
	@echo "✓ Build complete"

clean:
	rm -rf target

# ==============================================================================
# Run
# ==============================================================================

run: build check-txt
	./target/release/nomic-serve

run-full: build check-models
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
docker-build-cpu-full: model-txt-all model-img-all
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
docker-build-gpu-full: model-txt-all model-img-all
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
