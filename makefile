# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

.PHONY: model fmt build clean run health test test-list

# ==============================================================================
# Model Files
# ==============================================================================

tokenizer.json:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json

model_quantized.onnx:
	wget --content-disposition -q \
		https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx

model: tokenizer.json model_quantized.onnx
	@echo "✓ Model files ready"

# ==============================================================================
# Build
# ==============================================================================

fmt:
	cargo fmt

target/release/nomic-serve: src/main.rs Cargo.toml
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
	@curl -s http://localhost:8080/health && echo ""

test:
	@curl -s -X POST localhost:8080/embed \
		-H 'content-type: application/json' \
		-d '{"inputs": "ONNX in Rust is fast"}' | \
		jq '{tokens: .tokens[0], time_ms: (.time_ms | floor), dims: (.embeddings[0] | length), sample: (.embeddings[0][0:5] | map(. * 1000 | floor / 1000))}'

test-list:
	@curl -s -X POST localhost:8080/embed \
		-H 'content-type: application/json' \
		-d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
		jq '{count: (.embeddings | length), tokens, time_ms: (.time_ms | floor), samples: [.embeddings[] | .[0:3] | map(. * 1000 | floor / 1000)]}'
