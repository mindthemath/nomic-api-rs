#!/bin/bash
# Run model variant comparison tests
#
# Starts multiple server instances with different model variants,
# runs comparison tests, and cleans up on exit (including Ctrl-C).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BINARY="$PROJECT_ROOT/target/release/nomic-serve"
COMPARE_SCRIPT="$SCRIPT_DIR/compare_model_variants.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if binary exists
if [[ ! -f "$BINARY" ]]; then
    echo -e "${RED}❌ Binary not found: $BINARY${NC}"
    echo "Run 'make build' first"
    exit 1
fi

# Check if Python script exists
if [[ ! -f "$COMPARE_SCRIPT" ]]; then
    echo -e "${RED}❌ Comparison script not found: $COMPARE_SCRIPT${NC}"
    exit 1
fi

# Check Python dependencies
if ! python3 -c "import requests" 2>/dev/null; then
    echo -e "${YELLOW}⚠ Python 'requests' library not found${NC}"
    echo "  Install with: pip install requests"
    echo "  Or install all dependencies: pip install -r scripts/requirements.txt"
    exit 1
fi

# Check for GPU mode
USE_GPU="${USE_GPU:-0}"
if [[ "$USE_GPU" == "1" || "$USE_GPU" == "true" ]]; then
    echo -e "${GREEN}GPU mode enabled${NC}"
    GPU_ENV="USE_GPU=1"
    
    # Find ONNX Runtime CUDA providers library directory
    # The ort crate stores libraries in target/release/ or in ~/.cache/ort.pyke.io/
    ORT_LIB_DIR=""
    
    # Check target/release/deps first (most reliable)
    if [[ -d "$PROJECT_ROOT/target/release/deps" ]]; then
        # Find the actual library location via symlink
        if [[ -L "$PROJECT_ROOT/target/release/deps/libonnxruntime_providers_shared.so" ]]; then
            ORT_LIB_DIR=$(readlink -f "$PROJECT_ROOT/target/release/deps/libonnxruntime_providers_shared.so" | xargs dirname 2>/dev/null)
        elif [[ -f "$PROJECT_ROOT/target/release/deps/libonnxruntime_providers_shared.so" ]]; then
            ORT_LIB_DIR="$PROJECT_ROOT/target/release/deps"
        fi
    fi
    
    # Fallback to target/release/
    if [[ -z "$ORT_LIB_DIR" && -d "$PROJECT_ROOT/target/release" ]]; then
        if [[ -L "$PROJECT_ROOT/target/release/libonnxruntime_providers_shared.so" ]]; then
            ORT_LIB_DIR=$(readlink -f "$PROJECT_ROOT/target/release/libonnxruntime_providers_shared.so" | xargs dirname 2>/dev/null)
        elif [[ -f "$PROJECT_ROOT/target/release/libonnxruntime_providers_shared.so" ]]; then
            ORT_LIB_DIR="$PROJECT_ROOT/target/release"
        fi
    fi
    
    if [[ -n "$ORT_LIB_DIR" && -d "$ORT_LIB_DIR" ]]; then
        echo -e "  ${GREEN}Found ONNX Runtime libraries in: $ORT_LIB_DIR${NC}"
        export LD_LIBRARY_PATH="$ORT_LIB_DIR:${LD_LIBRARY_PATH:-}"
    else
        echo -e "  ${YELLOW}⚠ Warning: Could not find ONNX Runtime CUDA providers library${NC}"
        echo -e "  ${YELLOW}  GPU mode may fall back to CPU. Check logs for details.${NC}"
    fi
else
    GPU_ENV=""
fi

# Model configurations: (port, model_file, name)
# Baseline is fp32 (full precision, unquantized)
declare -a MODELS=(
    "8080:model.onnx:fp32 (baseline)"
    "8081:model_quantized.onnx:quantized"
    "8082:model_q4f16.onnx:q4f16"
    "8083:model_fp16.onnx:fp16"
)

# Track PIDs for cleanup
declare -a PIDS=()

# Cleanup function
cleanup() {
    echo -e "\n${YELLOW}Cleaning up servers...${NC}"
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping server (PID $pid)"
            kill "$pid" 2>/dev/null || true
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo -e "${GREEN}✓ Cleanup complete${NC}"
    exit 0
}

# Set trap for cleanup on exit
trap cleanup EXIT INT TERM

# Start servers
echo -e "${GREEN}Starting model servers...${NC}"
cd "$PROJECT_ROOT"

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r port model_file name <<< "$model_config"
    
    # Check if model file exists
    if [[ ! -f "$model_file" ]]; then
        echo -e "${YELLOW}⚠ Model file not found: $model_file (skipping)${NC}"
        continue
    fi
    
    echo "  Starting $name on port $port with $model_file"
    
    # Start server in background with optional GPU flag
    # LD_LIBRARY_PATH is already exported if GPU mode is enabled
    if [[ -n "$GPU_ENV" ]]; then
        env $GPU_ENV MODEL="$model_file" PORT="$port" "$BINARY" > "/tmp/nomic-serve-$port.log" 2>&1 &
    else
        env MODEL="$model_file" PORT="$port" "$BINARY" > "/tmp/nomic-serve-$port.log" 2>&1 &
    fi
    pid=$!
    PIDS+=("$pid")
    
    echo "    PID: $pid"
done

# Wait for servers to be ready
echo -e "\n${GREEN}Waiting for servers to be ready...${NC}"
sleep 2

max_attempts=30
for model_config in "${MODELS[@]}"; do
    IFS=':' read -r port model_file name <<< "$model_config"
    
    if [[ ! -f "$model_file" ]]; then
        continue
    fi
    
    attempt=0
    while ! curl -s "http://localhost:$port/health" > /dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [[ $attempt -ge $max_attempts ]]; then
            echo -e "${RED}❌ Server on port $port ($name) failed to start${NC}"
            echo "  Check logs: /tmp/nomic-serve-$port.log"
            exit 1
        fi
        sleep 0.5
    done
    echo -e "  ${GREEN}✓${NC} Port $port ($name) ready"
done

# Run comparisons
echo -e "\n${GREEN}Running model comparisons...${NC}"
if [[ -n "$GPU_ENV" ]]; then
    echo "Baseline: model.onnx (fp32, full precision, port 8080) [GPU]"
else
    echo "Baseline: model.onnx (fp32, full precision, port 8080) [CPU]"
fi
echo ""

# Compare each variant against baseline (fp32)
baseline_port=8080

for model_config in "${MODELS[@]}"; do
    IFS=':' read -r port model_file name <<< "$model_config"
    
    if [[ ! -f "$model_file" ]]; then
        continue
    fi
    
    if [[ $port -eq $baseline_port ]]; then
        continue  # Skip baseline itself
    fi
    
    echo -e "${YELLOW}Comparing $name (port $port) vs baseline...${NC}"
    python3 "$COMPARE_SCRIPT" "$baseline_port" "$port" "$name" || {
        echo -e "${RED}❌ Comparison failed for $name${NC}"
        continue
    }
done

echo -e "\n${GREEN}✓ All comparisons complete${NC}"

