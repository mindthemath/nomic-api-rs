tokenizer.json:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/tokenizer.json

model_quantized.onnx:
	wget --content-disposition https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main/onnx/model_quantized.onnx

fmt:
	cargo fmt

target/release/nomic-serve: src/main.rs
	cargo build --release

build: fmt target/release/nomic-serve

run:
	MODEL=model_quantized.onnx TOKENIZER=tokenizer.json ./target/release/nomic-serve	

health:
	curl -i http://localhost:8080/health

test:
	curl -s -X POST localhost:8080/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": "ONNX in Rust is fast" }' | \
     jq '{tokens: .tokens[0], time_ms, embedding_length: (.embeddings[0] | length), embeddings_sample: (.embeddings[0][0:5])}'

test-list:
	curl -s -X POST localhost:8080/embed \
     -H 'content-type: application/json' \
     -d '{"inputs": ["ONNX in Rust is fast", "Python is also great", "Embeddings are useful"]}' | \
     jq '{tokens, time_ms, count: (.embeddings | length), embedding_lengths: [.embeddings[] | length], samples: [.embeddings[] | .[0:3]]}'

clean:
	rm -rf target