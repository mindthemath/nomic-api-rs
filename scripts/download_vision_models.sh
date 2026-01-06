#!/usr/bin/env bash
# ==============================================================================
# Download nomic-embed-vision-v1.5 ONNX models
# https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/tree/main/onnx
# ==============================================================================

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-models/img}"
BASE_URL="https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5/resolve/main"

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

# Model variants
case "${1:-default}" in
    config)
        # Only download preprocessor config (for reference, not required at runtime)
        download "preprocessor_config.json" "$BASE_URL/preprocessor_config.json"
        ;;
    all)
        # Download preprocessor config + all model variants
        download "preprocessor_config.json" "$BASE_URL/preprocessor_config.json"
        download "model.onnx" "$BASE_URL/onnx/model.onnx"
        download "model_fp16.onnx" "$BASE_URL/onnx/model_fp16.onnx"
        download "model_quantized.onnx" "$BASE_URL/onnx/model_quantized.onnx"
        download "model_int8.onnx" "$BASE_URL/onnx/model_int8.onnx"
        download "model_uint8.onnx" "$BASE_URL/onnx/model_uint8.onnx"
        download "model_q4.onnx" "$BASE_URL/onnx/model_q4.onnx"
        download "model_bnb4.onnx" "$BASE_URL/onnx/model_bnb4.onnx"
        ;;
    fp32)
        download "model.onnx" "$BASE_URL/onnx/model.onnx"
        ;;
    fp16)
        download "model_fp16.onnx" "$BASE_URL/onnx/model_fp16.onnx"
        ;;
    q4)
        download "model_q4.onnx" "$BASE_URL/onnx/model_q4.onnx"
        ;;
    quantized|default|"")
        download "model_quantized.onnx" "$BASE_URL/onnx/model_quantized.onnx"
        ;;
    int8)
        download "model_int8.onnx" "$BASE_URL/onnx/model_int8.onnx"
        ;;
    uint8)
        download "model_uint8.onnx" "$BASE_URL/onnx/model_uint8.onnx"
        ;;
    bnb4)
        download "model_bnb4.onnx" "$BASE_URL/onnx/model_bnb4.onnx"
        ;;
    *)
        echo "Usage: $0 [default|all|fp32|fp16|quantized|int8|uint8|q4|bnb4|config]"
        echo ""
        echo "Variants:"
        echo "  default   - model_quantized.onnx (int8, recommended for CPU)"
        echo "  all       - all model variants + preprocessor_config.json"
        echo "  config    - preprocessor_config.json only (for reference)"
        echo "  fp32      - model.onnx (374 MB, full precision)"
        echo "  fp16      - model_fp16.onnx (187 MB, half precision, good for GPU)"
        echo "  quantized - model_quantized.onnx (97 MB, int8)"
        echo "  int8      - model_int8.onnx (97 MB, int8)"
        echo "  uint8     - model_uint8.onnx (97 MB, uint8)"
        echo "  q4        - model_q4.onnx (62 MB, 4-bit weights)"
        echo "  bnb4      - model_bnb4.onnx (56 MB, bitsandbytes 4-bit)"
        echo ""
        echo "Note: preprocessor_config.json is not required at runtime (constants are hardcoded in Rust)"
        exit 1
        ;;
esac

echo ""
echo "✓ Vision model files ready in $MODEL_DIR"

