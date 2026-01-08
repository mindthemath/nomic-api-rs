# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

default: run

.PHONY: model fmt build clean run health docs openapi test test-list test-dim models-all test-models \
        docker-build docker-build-cpu docker-push docker-push-cpu \
        model-txt model-txt-all model-img model-img-all check-txt check-img check-models \
        test-img test-img-batch test-multimodal test-img-stats run-stats \
        test-vision-batch test-vision-variants test-text-batch-fp32 test-text-batch-onnx-fp16 \
        test-fp16-accuracy test-text-batch-transformers test-text-batch-half-precision \
        benchmark-vision-batch benchmark-vision-batch-gpu benchmark-throughput

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
	cargo build --release

build: fmt target/release/nomic-serve
	@echo "✓ Build complete"

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
# Run
# ==============================================================================

run: build check-models
	./target/release/nomic-serve

run-full: build check-models
	AVERAGING=arithmetic TXT_MODEL=models/txt/model.onnx IMG_MODEL=models/img/model.onnx ./target/release/nomic-serve

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

# Test vision model batching safety (check for cross-sample interference)
test-vision-batch:
	@echo "Testing vision model batching for cross-sample interference..."
	@cd scripts && python3 test_vision_batch_interference.py

# Test text model FP32 vs quantized batching
test-text-batch-fp32:
	@echo "Testing text model batching: quantized vs FP32..."
	@cd scripts && python3 test_text_batch_fp32.py

# Test text model ONNX FP16 vs other variants
test-text-batch-onnx-fp16:
	@echo "Testing text model batching: ONNX FP16 vs INT8 vs FP32 vs Q4F16..."
	@cd scripts && python3 test_text_batch_onnx_fp16.py

# Test FP16 vs FP32 accuracy and batch sensitivity
test-fp16-accuracy:
	@echo "Testing FP16 vs FP32 accuracy and batch sensitivity..."
	@cd scripts && python3 test_fp16_vs_fp32_accuracy.py

# Test vision model variants (FP32, FP16, quantized)
test-vision-variants:
	@echo "Testing vision model variants: FP32 vs FP16 vs Quantized..."
	@cd scripts && python3 test_vision_model_variants.py

# Test text model with PyTorch/transformers
test-text-batch-transformers:
	@echo "Testing text model batching with PyTorch/transformers..."
	@cd scripts && python3 test_text_batch_transformers.py

# Test text model with PyTorch half-precision (FP16/BF16)
test-text-batch-half-precision:
	@echo "Testing text model batching with PyTorch half-precision (FP16/BF16)..."
	@cd scripts && python3 test_text_batch_half_precision.py

# Benchmark vision model batching performance
benchmark-vision-batch:
	@echo "Benchmarking vision model with different batch sizes..."
	@cd scripts && python3 benchmark_vision_batching.py

# Benchmark vision model on GPU (if available)
benchmark-vision-batch-gpu:
	@echo "Benchmarking vision model on GPU with different batch sizes..."
	@cd scripts && python3 benchmark_vision_batching.py --gpu

# Benchmark API server throughput
benchmark-throughput:
	@echo "Benchmarking API server throughput..."
	@echo "Make sure server is running: make run"
	@cd scripts && python3 benchmark_throughput.py

# Test Rust vision batching implementation via API
test-rust-batch:
	@echo "Testing Rust vision batching implementation..."
	@echo "Make sure server is running: make run"
	@cd scripts && python3 test_rust_batching.py

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
	@echo "Starting model variant comparison..."
	@bash scripts/run_model_comparison.sh

# ==============================================================================
# Docker
# ==============================================================================

DOCKER_IMAGE = mindthemath/nomic-embed-v1.5-rs
DOCKER_TAG ?= latest

# Build CPU image
docker-build: docker-build-cpu

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

# Push image
docker-push: docker-push-cpu

# Push CPU image
docker-push-cpu: docker-build-cpu
	@echo "Pushing CPU image to DockerHub..."
	docker push $(DOCKER_IMAGE):$(DOCKER_TAG)-cpu
	docker push $(DOCKER_IMAGE):latest-cpu
