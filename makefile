# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

.PHONY: model fmt build clean run health docs openapi test test-list test-dim models-all test-models \
        docker-build docker-build-cpu docker-build-gpu docker-push docker-push-cpu docker-push-gpu

# ==============================================================================
# Model Files
# ==============================================================================

tokenizer.json:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json

model_quantized.onnx:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx

model_q4f16.onnx:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_q4f16.onnx

model_fp16.onnx:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_fp16.onnx

model.onnx:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx

model: tokenizer.json model_quantized.onnx
	@echo "✓ Model files ready"

# Download all model variants for comparison
models-all: tokenizer.json model_quantized.onnx model_q4f16.onnx model_fp16.onnx model.onnx
	@echo "✓ All model variants downloaded"

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

run: build
	./target/release/nomic-serve

# ==============================================================================
# Test
# ==============================================================================

health:
	@curl -s http://localhost:8080/health | jq .

docs:
	@echo "Opening docs at http://localhost:8080/docs"
	@curl -s -o /dev/null -w "HTTP %{http_code}\n" http://localhost:8080/docs

openapi:
	@curl -s http://localhost:8080/openapi.json | jq '.info'

test:
	@curl -s -X POST localhost:8080/embed \
		-H 'content-type: application/json' \
		-d '{"inputs": "ONNX in Rust is fast"}' | \
		jq '{tokens: .tokens, time_ms: (.time_ms | floor), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

test-list:
	@curl -s -X POST localhost:8080/batch \
		-H 'content-type: application/json' \
		-d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
		jq '{count: (.embeddings | length), tokens, time_ms: (.time_ms | floor), samples: [.embeddings[] | .[0:3] | map(. * 1000 | floor / 1000)]}'

test-dim:
	@echo "Testing Matryoshka embeddings (dim=128)..."
	@curl -s -X POST localhost:8080/embed \
		-H 'content-type: application/json' \
		-d '{"inputs": "ONNX in Rust is fast", "dim": 128}' | \
		jq '{tokens: .tokens, time_ms: (.time_ms | floor), sample: (.embedding[0:5] | map(. * 1000 | floor / 1000))}'

# Compare all model variants against baseline (model.onnx, fp32)
# Requires: models-all, build, and Python requests library
test-models: build models-all
	@echo "Starting model variant comparison (CPU)..."
	@USE_GPU=0 bash scripts/run_model_comparison.sh

# Compare all model variants on GPU
# Requires: models-all, build, CUDA drivers, and Python requests library
test-models-gpu: build models-all
	@echo "Starting model variant comparison (GPU)..."
	@USE_GPU=1 bash scripts/run_model_comparison.sh

# ==============================================================================
# Docker
# ==============================================================================

DOCKER_IMAGE = mindthemath/nomic-text-v1.5-rs
DOCKER_TAG ?= latest

# Build both CPU and GPU images
docker-build: docker-build-cpu docker-build-gpu

# Build CPU-only image
docker-build-cpu: model
	@echo "Building CPU Docker image..."
	docker build --target runtime-cpu -t $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu -t $(DOCKER_IMAGE):latest-cpu .

docker-run-cpu: docker-build-cpu
	docker run -p 8080:8080 $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu

# Build GPU (CUDA) image
docker-build-gpu: model
	@echo "Building GPU Docker image..."
	docker build --target runtime-gpu -t $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu -t $(DOCKER_IMAGE):latest-gpu .

docker-run-gpu: docker-build-gpu
	docker run --gpus all -p 8080:8080 $(DOCKER_IMAGE):$(DOCKER_TAG)-gpu

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
