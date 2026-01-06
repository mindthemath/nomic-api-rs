# CLIP-Style Image Processing Pipeline

> How pixels travel from network data to embedding vectors in nomic-embed-vision-v1.5

This document traces every transformation an image undergoes from raw bytes to a 768-dimensional embedding vector. All claims are verified with Python scripts in `scripts/pipeline_demo.py`.

---

## Table of Contents

1. [Overview](#overview)
2. [Pipeline Stages](#pipeline-stages)
3. [Stage 1: Image Acquisition](#stage-1-image-acquisition)
4. [Stage 2: Decode to Pixels](#stage-2-decode-to-pixels)
5. [Stage 3: RGB Conversion](#stage-3-rgb-conversion)
6. [Stage 4: Resize](#stage-4-resize)
7. [Stage 5: Center Crop](#stage-5-center-crop)
8. [Stage 6: Rescale to Float](#stage-6-rescale-to-float)
9. [Stage 7: Normalize](#stage-7-normalize)
10. [Stage 8: Reshape to NCHW](#stage-8-reshape-to-nchw)
11. [Stage 9: ONNX Inference](#stage-9-onnx-inference)
12. [Stage 10: CLS Token Extraction](#stage-10-cls-token-extraction)
13. [Stage 11: L2 Normalization](#stage-11-l2-normalization)
14. [Stage 12: Dimension Truncation](#stage-12-dimension-truncation)
15. [Memory and Compute Analysis](#memory-and-compute-analysis)
16. [Why These Specific Values?](#why-these-specific-values)

---

## Overview

The **nomic-embed-vision-v1.5** model uses a **CLIP-style image processor** based on the Vision Transformer (ViT) architecture. CLIP (Contrastive Language-Image Pre-training) was developed by OpenAI and established preprocessing conventions that many vision models now follow.

**Key insight**: The model only ever "sees" a **224×224×3 = 150,528 float32 values** tensor, regardless of input image size. All preprocessing exists to transform arbitrary images into this fixed format.

```
┌─────────────────┐
│  Network Data   │  (JPEG/PNG bytes, URL, base64)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Decode Image   │  → Raw pixels (H×W×C, uint8)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Convert RGB    │  → Ensure 3 channels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Resize         │  → Shortest edge = 224px
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Center Crop    │  → 224×224 pixels
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Rescale        │  → [0, 255] → [0, 1]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Normalize      │  → (x - mean) / std
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Reshape NCHW   │  → [1, 3, 224, 224]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ViT Inference  │  → [1, 197, 768] hidden states
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  CLS Token      │  → [1, 768] raw embedding
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  L2 Normalize   │  → [1, 768] unit vector
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Truncate dim   │  → [1, dim] final embedding
└────────┬────────┘
```

---

## Pipeline Stages

### Stage 1: Image Acquisition

**Input**: URL, base64 string, or raw bytes  
**Output**: Raw image bytes

```python
# From URL
import requests
response = requests.get("https://example.com/image.jpg")
image_bytes = response.content

# From base64
import base64
image_bytes = base64.b64decode(base64_string)

# From data URL
# "data:image/png;base64,iVBORw0KGgo..."
_, encoded = data_url.split(",", 1)
image_bytes = base64.b64decode(encoded)
```

**Security considerations**:
- URL fetching should have timeouts and size limits
- Base64 decoding can expand data ~33% (4 base64 chars = 3 bytes)
- Consider validating Content-Type headers for URLs

---

### Stage 2: Decode to Pixels

**Input**: Raw image bytes (JPEG, PNG, WebP, GIF, etc.)  
**Output**: Pixel array, shape `(H, W, C)`, dtype `uint8`

```python
from PIL import Image
from io import BytesIO

image = Image.open(BytesIO(image_bytes))
# image.size = (width, height)
# image.mode = 'RGB', 'RGBA', 'L', 'P', etc.
```

**Format support**:
| Format | Typical Size | Notes |
|--------|-------------|-------|
| JPEG | Small | Lossy, no alpha, most photos |
| PNG | Medium | Lossless, supports alpha |
| WebP | Small | Modern, lossy or lossless |
| GIF | Variable | Animated (we use first frame) |

**Memory at this stage**: For a 4000×3000 photo = 12 megapixels × 3 bytes = **36 MB** raw pixels.

---

### Stage 3: RGB Conversion

**Input**: Image in any color mode  
**Output**: Image in RGB mode (3 channels)

```python
if image.mode != "RGB":
    image = image.convert("RGB")
```

**Common conversions**:
- `RGBA` → `RGB`: Discard alpha channel (composited onto white by default in PIL)
- `L` (grayscale) → `RGB`: Replicate single channel 3×
- `P` (palette) → `RGB`: Expand palette indices to RGB values
- `CMYK` → `RGB`: Color space conversion

**Why RGB?** The model was trained on RGB images. Using different color spaces would produce garbage embeddings.

---

### Stage 4: Resize

**Input**: RGB image of arbitrary size `(W, H)`  
**Output**: RGB image where shortest edge = 224 pixels

**Algorithm**: Resize such that the **shortest edge becomes 224**, maintaining aspect ratio.

```python
from PIL import Image

def resize_shortest_edge(image, target=224):
    """Resize so shortest edge = target, maintaining aspect ratio."""
    width, height = image.size
    
    if width < height:
        # Width is shorter
        new_width = target
        new_height = int(height * (target / width))
    else:
        # Height is shorter (or equal)
        new_height = target
        new_width = int(width * (target / height))
    
    return image.resize((new_width, new_height), Image.BICUBIC)
```

**Resampling method**: `BICUBIC` (Pillow's `resample=3`)
- Uses a 4×4 pixel neighborhood
- Smoother than bilinear, sharper than Lanczos
- Good balance of quality vs. speed

**Example transformations**:
| Input Size | After Resize | Notes |
|------------|--------------|-------|
| 640×480 | 299×224 | Height becomes 224 |
| 1920×1080 | 398×224 | Height becomes 224 |
| 480×640 | 224×299 | Width becomes 224 |
| 224×224 | 224×224 | No change needed |
| 100×100 | 224×224 | Upscaled |

**Important**: Images smaller than 224×224 get **upscaled**. This can reduce quality but is necessary for the model.

---

### Stage 5: Center Crop

**Input**: Resized image (one edge = 224, other ≥ 224)  
**Output**: 224×224 pixel image

```python
def center_crop(image, crop_size=224):
    """Extract center crop of crop_size × crop_size."""
    width, height = image.size
    
    left = (width - crop_size) // 2
    top = (height - crop_size) // 2
    right = left + crop_size
    bottom = top + crop_size
    
    return image.crop((left, top, right, bottom))
```

**What gets cropped?**
- For landscape images: left and right edges removed
- For portrait images: top and bottom edges removed
- For square images: nothing cropped

**Visual example** (landscape 400×224 → 224×224):
```
┌──────────────────────────────────────┐
│  cropped  │   KEPT (224px)   │ cropped │
│  (88px)   │                  │  (88px) │
└──────────────────────────────────────┘
```

**Limitation**: Content at edges may be lost. For some use cases (e.g., object detection), this matters. For semantic similarity, it usually doesn't.

---

### Stage 6: Rescale to Float

**Input**: 224×224 uint8 image, values in [0, 255]  
**Output**: 224×224 float32 image, values in [0.0, 1.0]

```python
import numpy as np

pixels = np.array(image, dtype=np.float32)
pixels = pixels * (1.0 / 255.0)  # or pixels / 255.0

# Shape: (224, 224, 3)
# Dtype: float32
# Range: [0.0, 1.0]
```

**Why 1/255?** The config specifies `rescale_factor: 0.00392156862745098`, which is exactly `1/255`.

**Memory**: 224 × 224 × 3 × 4 bytes = **602 KB** per image.

---

### Stage 7: Normalize

**Input**: Float pixels in [0.0, 1.0]  
**Output**: Normalized pixels (roughly [-2, +2] range)

**Formula**: `normalized = (pixel - mean) / std`

```python
# CLIP normalization constants (from preprocessor_config.json)
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

normalized = (pixels - MEAN) / STD
```

**What do these magic numbers mean?**

These are the **per-channel mean and standard deviation** computed from the dataset CLIP was trained on (400M image-text pairs from the internet).

| Channel | Mean | Std | Interpretation |
|---------|------|-----|----------------|
| Red | 0.481 | 0.269 | Average red ~49% brightness |
| Green | 0.458 | 0.261 | Average green ~46% brightness |
| Blue | 0.408 | 0.276 | Average blue ~41% brightness |

**Why normalize?**
1. **Zero-centered**: Helps gradients flow during training
2. **Unit variance**: Prevents some channels dominating others
3. **Consistency**: Model expects this exact distribution

**Resulting value range**:
- Black pixel `[0,0,0]` → `[-1.79, -1.75, -1.48]`
- White pixel `[1,1,1]` → `[1.93, 2.07, 2.14]`
- Typical range: approximately `[-2, +2]`

---

### Stage 8: Reshape to NCHW

**Input**: Normalized pixels, shape `(224, 224, 3)` (HWC format)  
**Output**: Tensor, shape `(1, 3, 224, 224)` (NCHW format)

```python
# HWC → CHW (channels first)
tensor = normalized.transpose(2, 0, 1)  # (3, 224, 224)

# Add batch dimension
tensor = np.expand_dims(tensor, axis=0)  # (1, 3, 224, 224)
```

**Why NCHW?**
- **N** = Batch size (1 for single image)
- **C** = Channels (3 for RGB)
- **H** = Height (224)
- **W** = Width (224)

This is the standard format for ONNX Runtime, PyTorch, and most deep learning frameworks. It's more efficient for GPU computation (contiguous memory access patterns for convolutions).

**Final input tensor**:
- Shape: `(1, 3, 224, 224)`
- Dtype: `float32`
- Memory: 150,528 floats × 4 bytes = **602 KB**

---

### Stage 9: ONNX Inference

**Input**: Tensor `(1, 3, 224, 224)`  
**Output**: Hidden states `(1, 197, 768)`

```python
import onnxruntime as ort

session = ort.InferenceSession("model.onnx")
outputs = session.run(None, {"pixel_values": tensor})
hidden_states = outputs[0]  # Shape: (1, 197, 768)
```

**What happens inside the model?**

The Vision Transformer (ViT) architecture:

1. **Patch Embedding**: Split 224×224 image into 14×14 = **196 patches** of 16×16 pixels each
2. **Linear Projection**: Each 16×16×3 = 768 pixel patch → 768-dim vector
3. **CLS Token**: Prepend a learnable "classification" token → now 197 tokens
4. **Position Embedding**: Add positional information to each token
5. **Transformer Blocks**: 12 layers of self-attention and feed-forward networks
6. **Output**: 197 tokens × 768 dimensions

```
Image (224×224×3)
       │
       ▼
┌─────────────────────────────────────────┐
│  Split into 14×14 = 196 patches         │
│  Each patch: 16×16×3 = 768 values       │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Linear projection: 768 → 768           │
│  Now: 196 patch tokens                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Prepend CLS token                      │
│  Now: 197 tokens (1 CLS + 196 patches)  │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  Add position embeddings                │
│  (tells model where each patch is)      │
└────────────────┬────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────┐
│  12× Transformer Blocks                 │
│  - Multi-head self-attention            │
│  - Feed-forward network                 │
│  - Layer normalization                  │
└────────────────┬────────────────────────┘
                 │
                 ▼
        Output: (1, 197, 768)
```

**Why 197 = 196 + 1?**
- 196 patch tokens (one per 16×16 region)
- 1 CLS (classification) token that aggregates global information

---

### Stage 10: CLS Token Extraction

**Input**: Hidden states `(1, 197, 768)`  
**Output**: Raw embedding `(1, 768)`

```python
cls_token = hidden_states[:, 0, :]  # First token is CLS
# Shape: (1, 768)
```

**Why the CLS token?**

During training, the model learns to aggregate image-wide information into the CLS token through self-attention. The 196 patch tokens contain **local** information; the CLS token contains **global** information.

**Alternative**: Mean pooling over all 197 tokens. Some models do this, but CLIP-style models use CLS only.

---

### Stage 11: L2 Normalization

**Input**: Raw embedding `(1, 768)`, arbitrary magnitude  
**Output**: Unit vector `(1, 768)`, L2 norm = 1.0

```python
import numpy as np

embedding = cls_token
l2_norm = np.linalg.norm(embedding, axis=-1, keepdims=True)
normalized_embedding = embedding / l2_norm

# Verify: np.linalg.norm(normalized_embedding) ≈ 1.0
```

**Why normalize?**

1. **Cosine similarity simplification**: For unit vectors, `cos(a,b) = dot(a,b)`. No need to divide by norms.
2. **Scale invariance**: Embedding magnitude doesn't affect similarity
3. **Numerical stability**: Bounded values prevent overflow

**Before vs After**:
```
Before: L2 norm ≈ 3000-4000 (varies by image)
After:  L2 norm = 1.0 (exactly)
```

---

### Stage 12: Dimension Truncation (Matryoshka)

**Input**: 768-dim embedding  
**Output**: `dim`-dimensional embedding (where `dim` ≤ 768)

```python
if dim < 768:
    embedding = embedding[:, :dim]
```

**Matryoshka embeddings**: The model was trained with Matryoshka Representation Learning, meaning the first N dimensions contain useful information even when truncated.

**Dimension vs Quality tradeoff**:
| Dimension | Storage | Quality | Use Case |
|-----------|---------|---------|----------|
| 768 | 100% | Best | High-precision retrieval |
| 512 | 67% | ~99% | Good default |
| 256 | 33% | ~97% | Mobile/embedded |
| 128 | 17% | ~94% | Fast approximate search |
| 64 | 8% | ~88% | Very constrained |

---

## Memory and Compute Analysis

### Memory Usage Per Image

| Stage | Size | Notes |
|-------|------|-------|
| Input bytes | Variable | JPEG ~100KB, PNG ~1MB typical |
| Decoded pixels | H×W×3 bytes | 12MP = 36MB |
| After resize | ~300×224×3 = 201KB | Depends on aspect ratio |
| After crop | 224×224×3 = 150KB | Fixed |
| Float32 tensor | 224×224×3×4 = 602KB | Fixed |
| Model inference | ~50MB | Internal activations |
| Output embedding | 768×4 = 3KB | Fixed |

**Peak memory**: Dominated by (1) decoded input image and (2) model inference.

### Why Image Size Limits Matter

A 100MP image (12000×8000):
- Decoded: 288 MB raw pixels
- Resize operation: Needs source + destination in memory
- Peak: ~500-600 MB for a single image

**Recommendation**: Limit input to 20MB compressed / ~50MP to prevent OOM.

### Compute Time (CPU, approximate)

| Stage | Time |
|-------|------|
| JPEG decode | 5-20ms |
| Resize | 2-10ms |
| Crop | <1ms |
| Normalize | <1ms |
| Model inference | 50-200ms |
| **Total** | **60-230ms** |

GPU inference is ~10× faster for the model, but preprocessing is CPU-bound regardless.

---

## Why These Specific Values?

### Image Size: 224×224

- **Historical**: AlexNet (2012) used 224×224, became standard
- **ViT patch math**: 224 ÷ 16 = 14 patches per side (clean division)
- **Memory**: Larger = quadratically more compute in self-attention

### Patch Size: 16×16

- **ViT-B/16**: The "16" means 16×16 patches
- **Balance**: Smaller patches = more tokens = more compute but finer detail
- **Standard**: 16×16 is the most common choice

### Hidden Dimension: 768

- **ViT-Base**: 768-dim hidden size
- **Matches BERT**: Enables multimodal alignment (text + image same dim)
- **ViT-Large** uses 1024, **ViT-Huge** uses 1280

### Normalization Constants

Computed empirically from CLIP's training data:
```python
# Approximate computation (pseudocode)
means = []
stds = []
for image in training_dataset:  # 400M images
    pixels = preprocess(image)  # [0, 1] range
    means.append(pixels.mean(axis=(0,1)))
    stds.append(pixels.std(axis=(0,1)))

MEAN = np.mean(means, axis=0)  # [0.481, 0.458, 0.408]
STD = np.mean(stds, axis=0)    # [0.269, 0.261, 0.276]
```

These are close to but not exactly ImageNet statistics, because CLIP's dataset differs.

---

## Appendix: Complete Python Reference

```python
"""
Complete image preprocessing pipeline for nomic-embed-vision-v1.5
"""
import numpy as np
from PIL import Image

# Constants from preprocessor_config.json
IMAGE_SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def preprocess_image(image: Image.Image) -> np.ndarray:
    """
    Preprocess image for nomic-embed-vision-v1.5.
    
    Args:
        image: PIL Image in any mode/size
        
    Returns:
        np.ndarray of shape (1, 3, 224, 224), dtype float32
    """
    # 1. Convert to RGB
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # 2. Resize (shortest edge to 224)
    w, h = image.size
    if w < h:
        new_w, new_h = IMAGE_SIZE, int(h * IMAGE_SIZE / w)
    else:
        new_w, new_h = int(w * IMAGE_SIZE / h), IMAGE_SIZE
    image = image.resize((new_w, new_h), Image.BICUBIC)
    
    # 3. Center crop to 224×224
    w, h = image.size
    left = (w - IMAGE_SIZE) // 2
    top = (h - IMAGE_SIZE) // 2
    image = image.crop((left, top, left + IMAGE_SIZE, top + IMAGE_SIZE))
    
    # 4. Convert to float32 [0, 1]
    pixels = np.array(image, dtype=np.float32) / 255.0
    
    # 5. Normalize
    pixels = (pixels - MEAN) / STD
    
    # 6. HWC → NCHW
    pixels = pixels.transpose(2, 0, 1)
    pixels = np.expand_dims(pixels, axis=0)
    
    return pixels


def postprocess_embedding(hidden_states: np.ndarray, dim: int = 768) -> np.ndarray:
    """
    Extract and normalize embedding from model output.
    
    Args:
        hidden_states: Model output, shape (1, 197, 768)
        dim: Output dimension (1-768)
        
    Returns:
        np.ndarray of shape (dim,), L2-normalized
    """
    # Extract CLS token
    cls_token = hidden_states[0, 0, :]  # Shape: (768,)
    
    # L2 normalize
    embedding = cls_token / np.linalg.norm(cls_token)
    
    # Truncate to requested dimension
    if dim < 768:
        embedding = embedding[:dim]
    
    return embedding
```

---

## References

1. [CLIP Paper](https://arxiv.org/abs/2103.00020) - Radford et al., 2021
2. [ViT Paper](https://arxiv.org/abs/2010.11929) - Dosovitskiy et al., 2020
3. [Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) - Kusupati et al., 2022
4. [nomic-embed-vision-v1.5](https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5) - Nomic AI
5. [preprocessor_config.json](../models/img/preprocessor_config.json) - Model configuration

