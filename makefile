# ==============================================================================
# nomic-serve Makefile
# ==============================================================================

# Download model files from HuggingFace
tokenizer.json:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json

model_quantized.onnx:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx

model_q4f16.onnx:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_q4f16.onnx

model_fp16.onnx:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_fp16.onnx

model.onnx:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model.onnx


# ==============================================================================
# Build
# ==============================================================================

fmt:
	cargo fmt

target/release/nomic-serve: src/main.rs
	cargo build --release

build: fmt target/release/nomic-serve

clean:
	rm -rf target

# ==============================================================================
# Run Server (different batch modes)
# ==============================================================================

# Default: NO_BATCH mode (sequential processing)
run:
	./target/release/nomic-serve

# NO_BATCH: Sequential processing (slowest, guaranteed correct)
run-nobatch:
	BATCH_MODE=NO_BATCH ./target/release/nomic-serve

# SAFE_BATCH: Group by token count, batch within groups (fast + exact)
run-safebatch:
	BATCH_MODE=SAFE_BATCH ./target/release/nomic-serve

# PAD_BATCH: Full batching with padding (fastest, slight differences)
run-padbatch:
	BATCH_MODE=PAD_BATCH ./target/release/nomic-serve

# ==============================================================================
# Test targets (hit port 8080)
# ==============================================================================

health:
	curl -i http://localhost:8080/health

# Single text embedding test
test:
	curl -s -X POST localhost:8080/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": "ONNX in Rust is fast" }' | \
     jq '{batch_mode, tokens: .tokens[0], time_ms, embedding_length: (.embeddings[0] | length), sample: (.embeddings[0][0:5])}'

# Multiple text embedding test
test-list:
	curl -s -X POST localhost:8080/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
     jq '{batch_mode, tokens, time_ms, count: (.embeddings | length), samples: [.embeddings[] | .[0:3]]}'

# ==============================================================================
# Test targets for alternate port (for running tests independently)
# ==============================================================================

# Run server on port 8081 with specified batch mode
run-test-nobatch:
	PORT=8081 BATCH_MODE=NO_BATCH ./target/release/nomic-serve

run-test-safebatch:
	PORT=8081 BATCH_MODE=SAFE_BATCH ./target/release/nomic-serve

run-test-padbatch:
	PORT=8081 BATCH_MODE=PAD_BATCH ./target/release/nomic-serve

# Test against port 8081
test-alt:
	curl -s -X POST localhost:8081/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
     jq '{batch_mode, tokens, time_ms, count: (.embeddings | length), samples: [.embeddings[] | .[0:3]]}'

# ==============================================================================
# Comparison tests
# ==============================================================================

# Compare two servers (8080 vs 8081) - run with different batch modes
compare:
	@echo "=== Server on PORT 8080 ===" 
	@curl -s -X POST localhost:8080/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
     jq '{batch_mode, samples: [.embeddings[] | .[0:3]]}'
	@echo ""
	@echo "=== Server on PORT 8081 ==="
	@curl -s -X POST localhost:8081/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
     jq '{batch_mode, samples: [.embeddings[] | .[0:3]]}'

# Verify SAFE_BATCH matches NO_BATCH exactly
verify-safebatch:
	@echo "Testing that SAFE_BATCH produces identical results to NO_BATCH..."
	@echo ""
	@NOBATCH=$$(curl -s -X POST localhost:8080/embed -H 'content-type: application/json' \
		-d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | jq -c '.embeddings'); \
	SAFEBATCH=$$(curl -s -X POST localhost:8081/embed -H 'content-type: application/json' \
		-d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | jq -c '.embeddings'); \
	if [ "$$NOBATCH" = "$$SAFEBATCH" ]; then \
		echo "✅ PASS: Results are identical"; \
	else \
		echo "❌ FAIL: Results differ"; \
		echo "NO_BATCH:   $$(echo $$NOBATCH | jq '.[0][0:3]')"; \
		echo "SAFE_BATCH: $$(echo $$SAFEBATCH | jq '.[0][0:3]')"; \
	fi

.PHONY: fmt build clean run run-nobatch run-safebatch run-padbatch \
        health test test-list \
        run-test-nobatch run-test-safebatch run-test-padbatch test-alt \
        compare verify-safebatch
