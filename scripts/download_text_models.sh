#!/usr/bin/env bash
# ==============================================================================
# Download nomic-embed-text-v1.5 ONNX models
# https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/tree/main/onnx
# ==============================================================================

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/txt}"
BASE_URL="https://huggingface.co/nomic-ai/nomic-embed-text-v1.5/resolve/main"

mkdir -p "$MODEL_DIR"

download() {
    local filename="$1"
    local url="$2"
    local dest="$MODEL_DIR/$filename"
    
    if [[ -f "$dest" ]]; then
        echo "✓ $filename already exists"
        return 0
    fi
    
    echo "⬇ Downloading $filename..."
    wget -q --show-progress -O "$dest" "$url"
    echo "✓ $filename downloaded"
}

# Tokenizer (required)
download "tokenizer.json" "$BASE_URL/tokenizer.json"

# Model variants
case "${1:-default}" in
    all)
        download "model.onnx" "$BASE_URL/onnx/model.onnx"
        download "model_fp16.onnx" "$BASE_URL/onnx/model_fp16.onnx"
        download "model_quantized.onnx" "$BASE_URL/onnx/model_quantized.onnx"
        download "model_q4f16.onnx" "$BASE_URL/onnx/model_q4f16.onnx"
        ;;
    fp32)
        download "model.onnx" "$BASE_URL/onnx/model.onnx"
        ;;
    fp16)
        download "model_fp16.onnx" "$BASE_URL/onnx/model_fp16.onnx"
        ;;
    q4f16)
        download "model_q4f16.onnx" "$BASE_URL/onnx/model_q4f16.onnx"
        ;;
    quantized|default|"")
        download "model_quantized.onnx" "$BASE_URL/onnx/model_quantized.onnx"
        ;;
    *)
        echo "Usage: $0 [default|all|fp32|fp16|quantized|q4f16]"
        echo ""
        echo "Variants:"
        echo "  default   - model_quantized.onnx (int8, recommended for CPU)"
        echo "  all       - all model variants"
        echo "  fp32      - model.onnx (full precision)"
        echo "  fp16      - model_fp16.onnx (half precision, good for GPU)"
        echo "  quantized - model_quantized.onnx (int8)"
        echo "  q4f16     - model_q4f16.onnx (4-bit weights)"
        exit 1
        ;;
esac

echo ""
echo "✓ Text model files ready in $MODEL_DIR"

