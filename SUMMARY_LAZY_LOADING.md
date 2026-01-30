# Lazy Model Downloading Implementation Summary

This document summarizes the changes implemented to transition from "Fat Images" (with baked-in models) to "Slim Images" (with on-demand model downloading).

## 🚀 Architectural Shift
Previously, Docker images were variant-specific (quantized vs. full) and contained 100MB-500MB of model data. The new architecture uses a single slim image that fetches the required models from HuggingFace only when their respective endpoints are accessed.

### 1. Backend Implementation (`src/main.rs`)
- **Lazy Initialization**: Replaced immediate model loading with thread-safe `tokio::sync::OnceCell`.
- **On-Demand Downloading**: Implemented `ensure_model_exists` which maps `TXT_MODEL` and `IMG_MODEL` environment variables to official HuggingFace URLs.
- **Dynamic Session Building**: Session builders are now created at runtime, supporting both CPU and GPU (if requested and available).
- **Correctness Enforced**: Maintained the strict batching restriction for quantized models (rejections with 400 Bad Request) while allowing full batching for FP32 models.

### 2. Docker & CI/CD Efficiency
- **Image Size Reduction**: 
    - **Before**: ~1.06GB (CPU Full)
    - **After**: **142MB** (CPU)
    - *Reduction of ~87%*
- **Unified Tagging**: Removed variant-specific tags. We now only publish `cpu`, `gpu`, and `latest`.
- **Flexible Runtime**: Users can switch between `model_quantized.onnx`, `model.onnx`, or `model_fp16.onnx` simply by changing an environment variable, without pulling a new image.

### 3. Developer Experience (`makefile` & `README.md`)
- **Simplified Workflow**: `make run` no longer requires running `make model-txt` first; it handles it automatically on the first request.
- **Improved Documentation**: Updated `README.md` with:
    - Clear instructions on volume mounting (`-v nomic-models:/app/models`) for persistence.
    - Updated configuration table reflecting that `TXT_MODEL`/`IMG_MODEL` now act as both local paths and remote identifiers.
    - Standardized API parameters (`input`/`inputs` across all endpoints).

## 🛠 Verification Results
- **Cold Start**: Confirmed that hit a text endpoint triggers a download + initialization (~10-30s depending on network).
- **Warm Start**: Subsequent requests take ~25-35ms.
- **Health Check**: Confirmed `/health` correctly reports `false` before download and `true` after.
- **Persistence**: Verified that mounting a Docker volume allows models to survive container restarts.

## ⚠️ Critical Review Notes
- **Internet Requirement**: Containers now require internet access on their first run to fetch models. For air-gapped environments, users should pre-download models and mount them to `/app/models`.
- **Disk Usage**: Ensure the host has enough space for the requested models (~100MB-900MB depending on selection).
