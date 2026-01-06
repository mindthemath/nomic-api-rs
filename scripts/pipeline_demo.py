#!/usr/bin/env python3
"""
Image Processing Pipeline Demo for nomic-embed-vision-v1.5

This script traces through every stage of the image processing pipeline,
printing intermediate values and verifying the documentation claims.

Usage:
    python scripts/pipeline_demo.py [image_path_or_url]

If no image is provided, creates a synthetic test image.
"""

import json
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

# Check dependencies
try:
    import onnxruntime as ort
except ImportError:
    print("Missing: pip install onnxruntime")
    sys.exit(1)

# ============================================================================
# Constants from preprocessor_config.json
# ============================================================================

IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2  # 196
NUM_TOKENS = NUM_PATCHES + 1  # 197 (patches + CLS)
HIDDEN_DIM = 768

MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def print_header(stage: int, name: str):
    """Print a stage header."""
    print(f"\n{'=' * 70}")
    print(f"STAGE {stage}: {name}")
    print("=" * 70)


def print_tensor_info(name: str, arr: np.ndarray, show_sample: bool = True):
    """Print tensor shape, dtype, and sample values."""
    print(f"\n{name}:")
    print(f"  Shape: {arr.shape}")
    print(f"  Dtype: {arr.dtype}")
    print(f"  Range: [{arr.min():.4f}, {arr.max():.4f}]")
    print(f"  Mean:  {arr.mean():.4f}")
    if show_sample and arr.size > 5:
        flat = arr.flatten()
        print(f"  First 5 values: {flat[:5]}")


def create_test_image(
    width: int = 640, height: int = 480, seed: int = 42
) -> Image.Image:
    """Create a reproducible synthetic test image."""
    np.random.seed(seed)

    # Create gradient + noise pattern
    r = np.tile(np.linspace(0, 255, width, dtype=np.uint8), (height, 1))
    g = np.tile(np.linspace(0, 255, height, dtype=np.uint8).reshape(-1, 1), (1, width))
    b = np.random.randint(0, 256, (height, width), dtype=np.uint8)

    pixels = np.stack([r, g, b], axis=-1)
    return Image.fromarray(pixels, mode="RGB")


def load_image(source: str) -> tuple[Image.Image, bytes]:
    """Load image from path or URL, return Image and original bytes."""
    if source.startswith(("http://", "https://")):
        import requests

        print(f"Fetching from URL: {source}")
        response = requests.get(source, timeout=30)
        response.raise_for_status()
        image_bytes = response.content
        image = Image.open(BytesIO(image_bytes))
    elif Path(source).exists():
        print(f"Loading from file: {source}")
        with open(source, "rb") as f:
            image_bytes = f.read()
        image = Image.open(BytesIO(image_bytes))
    else:
        raise ValueError(f"Cannot load image: {source}")

    return image, image_bytes


def main():
    print("=" * 70)
    print("CLIP-Style Image Processing Pipeline Demo")
    print("Model: nomic-embed-vision-v1.5")
    print("=" * 70)

    # ========================================================================
    # STAGE 1: Image Acquisition
    # ========================================================================
    print_header(1, "Image Acquisition")

    if len(sys.argv) > 1:
        image, image_bytes = load_image(sys.argv[1])
        print(f"  Loaded from: {sys.argv[1]}")
    else:
        print("  Creating synthetic test image (640x480)")
        image = create_test_image(640, 480)
        # Simulate "bytes" by encoding to JPEG
        buffer = BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        image_bytes = buffer.getvalue()

    print(
        f"  Compressed size: {len(image_bytes):,} bytes ({len(image_bytes)/1024:.1f} KB)"
    )

    # ========================================================================
    # STAGE 2: Decode to Pixels
    # ========================================================================
    print_header(2, "Decode to Pixels")

    # Re-decode from bytes to simulate full pipeline
    decoded_image = Image.open(BytesIO(image_bytes))

    print(f"  Image size: {decoded_image.size} (width × height)")
    print(f"  Image mode: {decoded_image.mode}")
    print(f"  Pixel count: {decoded_image.size[0] * decoded_image.size[1]:,}")

    raw_pixels = np.array(decoded_image)
    print(f"  Raw array shape: {raw_pixels.shape}")
    print(f"  Raw array dtype: {raw_pixels.dtype}")
    print(f"  Memory: {raw_pixels.nbytes:,} bytes ({raw_pixels.nbytes/1024:.1f} KB)")

    # ========================================================================
    # STAGE 3: RGB Conversion
    # ========================================================================
    print_header(3, "RGB Conversion")

    original_mode = decoded_image.mode
    if decoded_image.mode != "RGB":
        rgb_image = decoded_image.convert("RGB")
        print(f"  Converted: {original_mode} → RGB")
    else:
        rgb_image = decoded_image
        print(f"  Already RGB, no conversion needed")

    rgb_pixels = np.array(rgb_image)
    print(f"  Shape after RGB: {rgb_pixels.shape}")

    # ========================================================================
    # STAGE 4: Resize
    # ========================================================================
    print_header(4, "Resize (Shortest Edge → 224)")

    w, h = rgb_image.size
    print(f"  Input size: {w} × {h}")

    if w < h:
        new_w = IMAGE_SIZE
        new_h = int(h * IMAGE_SIZE / w)
        print(f"  Width is shorter edge")
    else:
        new_h = IMAGE_SIZE
        new_w = int(w * IMAGE_SIZE / h)
        print(f"  Height is shorter edge")

    resized_image = rgb_image.resize((new_w, new_h), Image.BICUBIC)
    print(f"  Output size: {new_w} × {new_h}")
    print(f"  Resampling: BICUBIC (high quality)")

    resized_pixels = np.array(resized_image)
    print_tensor_info("Resized pixels", resized_pixels)

    # ========================================================================
    # STAGE 5: Center Crop
    # ========================================================================
    print_header(5, f"Center Crop ({IMAGE_SIZE}×{IMAGE_SIZE})")

    w, h = resized_image.size
    left = (w - IMAGE_SIZE) // 2
    top = (h - IMAGE_SIZE) // 2
    right = left + IMAGE_SIZE
    bottom = top + IMAGE_SIZE

    print(f"  Input size: {w} × {h}")
    print(f"  Crop box: left={left}, top={top}, right={right}, bottom={bottom}")

    cropped_image = resized_image.crop((left, top, right, bottom))
    print(f"  Output size: {cropped_image.size[0]} × {cropped_image.size[1]}")

    if w > IMAGE_SIZE:
        print(f"  Cropped {(w - IMAGE_SIZE) // 2}px from each side (horizontal)")
    if h > IMAGE_SIZE:
        print(f"  Cropped {(h - IMAGE_SIZE) // 2}px from top/bottom (vertical)")

    cropped_pixels = np.array(cropped_image)
    print_tensor_info("Cropped pixels (uint8)", cropped_pixels)

    # ========================================================================
    # STAGE 6: Rescale to Float
    # ========================================================================
    print_header(6, "Rescale to Float [0, 1]")

    float_pixels = cropped_pixels.astype(np.float32) / 255.0

    print(f"  Formula: pixel / 255.0")
    print(f"  Example: uint8 value 128 → {128/255.0:.6f}")
    print_tensor_info("Float pixels [0,1]", float_pixels)

    # Show per-channel statistics
    print("\n  Per-channel means (before normalization):")
    for i, name in enumerate(["Red", "Green", "Blue"]):
        channel_mean = float_pixels[:, :, i].mean()
        print(f"    {name}: {channel_mean:.4f}")

    # ========================================================================
    # STAGE 7: Normalize
    # ========================================================================
    print_header(7, "Normalize with CLIP Statistics")

    print(f"  Formula: (pixel - mean) / std")
    print(f"  Mean: {MEAN}")
    print(f"  Std:  {STD}")

    normalized_pixels = (float_pixels - MEAN) / STD

    print("\n  Example transformations:")
    print(
        f"    Black [0,0,0] → [{-MEAN[0]/STD[0]:.3f}, {-MEAN[1]/STD[1]:.3f}, {-MEAN[2]/STD[2]:.3f}]"
    )
    print(
        f"    White [1,1,1] → [{(1-MEAN[0])/STD[0]:.3f}, {(1-MEAN[1])/STD[1]:.3f}, {(1-MEAN[2])/STD[2]:.3f}]"
    )

    print_tensor_info("Normalized pixels", normalized_pixels)

    print("\n  Per-channel statistics after normalization:")
    for i, name in enumerate(["Red", "Green", "Blue"]):
        ch = normalized_pixels[:, :, i]
        print(
            f"    {name}: mean={ch.mean():.4f}, std={ch.std():.4f}, range=[{ch.min():.3f}, {ch.max():.3f}]"
        )

    # ========================================================================
    # STAGE 8: Reshape to NCHW
    # ========================================================================
    print_header(8, "Reshape to NCHW Tensor")

    print(f"  Input shape (HWC): {normalized_pixels.shape}")

    # HWC → CHW
    chw_tensor = normalized_pixels.transpose(2, 0, 1)
    print(f"  After transpose (CHW): {chw_tensor.shape}")

    # Add batch dimension
    nchw_tensor = np.expand_dims(chw_tensor, axis=0)
    print(f"  After batch dim (NCHW): {nchw_tensor.shape}")

    print(
        f"\n  Memory: {nchw_tensor.nbytes:,} bytes ({nchw_tensor.nbytes/1024:.1f} KB)"
    )
    print(f"  This is the MODEL INPUT tensor")

    # ========================================================================
    # STAGE 9: ONNX Inference
    # ========================================================================
    print_header(9, "ONNX Model Inference")

    model_path = Path("models/img/model_quantized.onnx")
    if not model_path.exists():
        model_path = Path("models/img/model.onnx")

    if not model_path.exists():
        print(f"  ⚠️ Model not found. Run: bash scripts/download_vision_models.sh")
        return

    print(f"  Loading model: {model_path}")
    session = ort.InferenceSession(str(model_path))

    # Print model I/O info
    print(f"\n  Model inputs:")
    for inp in session.get_inputs():
        print(f"    {inp.name}: {inp.shape} ({inp.type})")

    print(f"\n  Model outputs:")
    for out in session.get_outputs():
        print(f"    {out.name}: {out.shape} ({out.type})")

    # Run inference
    print(f"\n  Running inference...")
    start_time = time.perf_counter()
    outputs = session.run(None, {"pixel_values": nchw_tensor})
    inference_time = (time.perf_counter() - start_time) * 1000

    hidden_states = outputs[0]
    print(f"  Inference time: {inference_time:.1f} ms")
    print_tensor_info("Hidden states output", hidden_states, show_sample=False)

    print(f"\n  Hidden states breakdown:")
    print(f"    Batch size: {hidden_states.shape[0]}")
    print(f"    Num tokens: {hidden_states.shape[1]} (1 CLS + {NUM_PATCHES} patches)")
    print(f"    Hidden dim: {hidden_states.shape[2]}")

    # ========================================================================
    # STAGE 10: CLS Token Extraction
    # ========================================================================
    print_header(10, "CLS Token Extraction")

    cls_token = hidden_states[0, 0, :]  # First batch, first token

    print(f"  Indexing: hidden_states[0, 0, :]")
    print(f"  CLS is the FIRST token (index 0)")
    print_tensor_info("CLS token (raw)", cls_token)

    # Show patch tokens too
    patch_tokens = hidden_states[0, 1:, :]
    print(f"\n  Patch tokens shape: {patch_tokens.shape}")
    print(f"  Patch tokens are indices 1-{NUM_TOKENS-1}")

    # ========================================================================
    # STAGE 11: L2 Normalization
    # ========================================================================
    print_header(11, "L2 Normalization")

    l2_norm_before = np.linalg.norm(cls_token)
    print(f"  L2 norm before: {l2_norm_before:.4f}")

    normalized_embedding = cls_token / l2_norm_before

    l2_norm_after = np.linalg.norm(normalized_embedding)
    print(f"  L2 norm after: {l2_norm_after:.6f}")

    print(f"\n  Formula: embedding / ||embedding||₂")
    print_tensor_info("Normalized embedding", normalized_embedding)

    # ========================================================================
    # STAGE 12: Dimension Truncation (Matryoshka)
    # ========================================================================
    print_header(12, "Dimension Truncation (Matryoshka)")

    print(f"  Full embedding: {len(normalized_embedding)} dimensions")

    for dim in [768, 512, 256, 128, 64]:
        truncated = normalized_embedding[:dim]
        # Re-normalize after truncation for fair comparison
        truncated_norm = truncated / np.linalg.norm(truncated)
        print(
            f"  dim={dim:3d}: first 3 values = [{truncated_norm[0]:.4f}, {truncated_norm[1]:.4f}, {truncated_norm[2]:.4f}]"
        )

    print(
        "\n Note: due to normalization, the first elements of the Matryoshka embeddings are not the same as each other."
    )
    # ========================================================================
    # Final Summary
    # ========================================================================
    print("\n" + "=" * 70)
    print("FINAL EMBEDDING")
    print("=" * 70)
    print(f"\n  Shape: ({HIDDEN_DIM},)")
    print(f"  Dtype: float32")
    print(f"  L2 norm: {np.linalg.norm(normalized_embedding):.6f}")
    print(f"  First 10 values:")
    print(f"    {normalized_embedding[:10]}")

    print("\n" + "=" * 70)
    print("PIPELINE SUMMARY")
    print("=" * 70)
    print(
        f"""
  Input:  {image.size[0]}×{image.size[1]} image ({len(image_bytes):,} bytes compressed)
  Output: {HIDDEN_DIM}-dimensional unit vector
  
  Transformations:
    1. Decode bytes → {raw_pixels.shape} uint8 array
    2. RGB convert → 3 channels
    3. Resize → {resized_image.size[0]}×{resized_image.size[1]} (shortest=224)
    4. Center crop → 224×224
    5. Rescale → float32 [0,1]
    6. Normalize → ~[-2, +2] range (CLIP statistics)
    7. Reshape → (1, 3, 224, 224) NCHW tensor
    8. ViT inference → (1, 197, 768) hidden states
    9. Extract CLS → (768,) raw embedding
   10. L2 normalize → (768,) unit vector
   
  The model only ever sees 224×224×3 = 150,528 float values.
  Everything else is discarded.
"""
    )


if __name__ == "__main__":
    main()
