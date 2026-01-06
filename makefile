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
     jq '{tokens, time_ms, embedding_length: (.embeddings | length), embeddings_sample: (.embeddings[0:5])}'

clean:
	rm -rf target