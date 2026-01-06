#!/usr/bin/env python3
"""
Test nomic-embed-vision-v1.5 ONNX model preprocessing.

Verifies:
1. Correct image preprocessing pipeline (CLIP-style)
2. Expected input tensor shape
3. Expected output embedding shape
4. Comparison with HuggingFace transformers implementation (if available)
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# Check for onnxruntime
try:
    import onnxruntime as ort
except ImportError:
    print("Missing: pip install onnxruntime")
    sys.exit(1)

MODEL_DIR = Path("models/img")
CONFIG_PATH = MODEL_DIR / "preprocessor_config.json"
MODEL_PATH = MODEL_DIR / "model_quantized.onnx"


def load_config():
    """Load preprocessor config."""
    with open(CONFIG_PATH) as f:
        return json.load(f)


def preprocess_image_manual(image: Image.Image, config: dict) -> np.ndarray:
    """
    Manually preprocess image following CLIPImageProcessor config.
    
    Pipeline:
    1. Convert to RGB
    2. Resize to size (224x224) with BICUBIC resampling
    3. Center crop to crop_size (224x224)
    4. Rescale by rescale_factor (1/255)
    5. Normalize with image_mean and image_std
    6. Convert to CHW format
    """
    # 1. Convert to RGB
    if config.get("do_convert_rgb", True):
        image = image.convert("RGB")
    
    # 2. Resize
    if config.get("do_resize", True):
        size = config["size"]
        # resample=3 is BICUBIC in PIL
        resample_map = {0: Image.NEAREST, 1: Image.BILINEAR, 2: Image.BILINEAR, 3: Image.BICUBIC}
        resample = resample_map.get(config.get("resample", 3), Image.BICUBIC)
        
        # For CLIP, resize such that shortest edge matches target, then center crop
        # But if size is a dict with both dimensions, resize to that directly
        if isinstance(size, dict):
            target_h, target_w = size["height"], size["width"]
        else:
            target_h = target_w = size
            
        # Resize maintaining aspect ratio with shortest edge = target
        w, h = image.size
        if w < h:
            new_w = target_w
            new_h = int(h * (target_w / w))
        else:
            new_h = target_h
            new_w = int(w * (target_h / h))
        
        image = image.resize((new_w, new_h), resample)
    
    # 3. Center crop
    if config.get("do_center_crop", True):
        crop_size = config["crop_size"]
        if isinstance(crop_size, dict):
            crop_h, crop_w = crop_size["height"], crop_size["width"]
        else:
            crop_h = crop_w = crop_size
            
        w, h = image.size
        left = (w - crop_w) // 2
        top = (h - crop_h) // 2
        right = left + crop_w
        bottom = top + crop_h
        image = image.crop((left, top, right, bottom))
    
    # Convert to numpy array (HWC, uint8)
    pixels = np.array(image, dtype=np.float32)
    
    # 4. Rescale (divide by 255)
    if config.get("do_rescale", True):
        rescale_factor = config.get("rescale_factor", 1/255)
        pixels = pixels * rescale_factor
    
    # 5. Normalize
    if config.get("do_normalize", True):
        mean = np.array(config["image_mean"], dtype=np.float32)
        std = np.array(config["image_std"], dtype=np.float32)
        pixels = (pixels - mean) / std
    
    # 6. Convert HWC -> CHW and add batch dimension -> NCHW
    pixels = pixels.transpose(2, 0, 1)  # HWC -> CHW
    pixels = np.expand_dims(pixels, axis=0)  # Add batch dim -> NCHW
    
    return pixels.astype(np.float32)


def test_with_transformers(image: Image.Image):
    """Test with HuggingFace transformers for comparison (if available)."""
    try:
        from transformers import AutoImageProcessor
        processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
        inputs = processor(images=image, return_tensors="np")
        return inputs["pixel_values"]
    except ImportError:
        print("  (transformers not installed, skipping comparison)")
        return None
    except Exception as e:
        print(f"  (transformers comparison failed: {e})")
        return None


def create_test_image(size=(640, 480), seed=42):
    """Create a reproducible test image."""
    np.random.seed(seed)
    # Create a simple gradient with some noise
    h, w = size[1], size[0]
    r = np.linspace(0, 255, w, dtype=np.uint8)
    g = np.linspace(0, 255, h, dtype=np.uint8)
    r = np.tile(r, (h, 1))
    g = np.tile(g.reshape(-1, 1), (1, w))
    b = np.random.randint(0, 256, (h, w), dtype=np.uint8)
    
    pixels = np.stack([r, g, b], axis=-1)
    return Image.fromarray(pixels, mode="RGB")


def main():
    print("=" * 70)
    print("nomic-embed-vision-v1.5 Preprocessing Test")
    print("=" * 70)
    
    # Load config
    if not CONFIG_PATH.exists():
        print(f"❌ Config not found: {CONFIG_PATH}")
        print("Run: bash scripts/download_vision_models.sh")
        sys.exit(1)
    
    config = load_config()
    print(f"\n📋 Preprocessor Config:")
    print(f"  - Resize to: {config['size']}")
    print(f"  - Center crop to: {config['crop_size']}")
    print(f"  - Rescale factor: {config['rescale_factor']}")
    print(f"  - Image mean: {config['image_mean']}")
    print(f"  - Image std: {config['image_std']}")
    print(f"  - Resample mode: {config['resample']} (3=BICUBIC)")
    
    # Create test image
    print(f"\n🖼️  Creating test image...")
    test_image = create_test_image(size=(640, 480))
    print(f"  - Original size: {test_image.size}")
    
    # Manual preprocessing
    print(f"\n🔧 Manual preprocessing...")
    manual_tensor = preprocess_image_manual(test_image, config)
    print(f"  - Output shape: {manual_tensor.shape}")
    print(f"  - Dtype: {manual_tensor.dtype}")
    print(f"  - Value range: [{manual_tensor.min():.4f}, {manual_tensor.max():.4f}]")
    print(f"  - Mean per channel: {manual_tensor.mean(axis=(0, 2, 3))}")
    
    # Compare with transformers (if available)
    print(f"\n🔬 Comparing with HuggingFace transformers...")
    hf_tensor = test_with_transformers(test_image)
    if hf_tensor is not None:
        print(f"  - HF output shape: {hf_tensor.shape}")
        print(f"  - HF value range: [{hf_tensor.min():.4f}, {hf_tensor.max():.4f}]")
        
        # Compare
        diff = np.abs(manual_tensor - hf_tensor)
        print(f"  - Max absolute diff: {diff.max():.6f}")
        print(f"  - Mean absolute diff: {diff.mean():.6f}")
        
        if diff.max() < 1e-4:
            print("  ✅ Manual preprocessing matches HuggingFace!")
        else:
            print("  ⚠️  Preprocessing differs from HuggingFace")
    
    # Test model inference
    if MODEL_PATH.exists():
        print(f"\n🧠 Testing ONNX model inference...")
        session = ort.InferenceSession(str(MODEL_PATH))
        
        # Get input/output info
        print(f"  Inputs:")
        for inp in session.get_inputs():
            print(f"    - {inp.name}: {inp.shape} ({inp.type})")
        
        print(f"  Outputs:")
        for out in session.get_outputs():
            print(f"    - {out.name}: {out.shape} ({out.type})")
        
        # Run inference
        input_name = session.get_inputs()[0].name
        outputs = session.run(None, {input_name: manual_tensor})
        
        embedding = outputs[0]
        print(f"\n📊 Embedding result:")
        print(f"  - Shape: {embedding.shape}")
        print(f"  - First 5 values: {embedding[0, :5]}")
        print(f"  - L2 norm: {np.linalg.norm(embedding):.4f}")
        
        # Check if it needs normalization
        if abs(np.linalg.norm(embedding) - 1.0) > 0.01:
            print(f"  ⚠️  Embedding is NOT unit normalized (L2 norm != 1)")
            print(f"  → Will need to L2-normalize in post-processing")
        else:
            print(f"  ✅ Embedding is unit normalized")
    else:
        print(f"\n⚠️  Model not found: {MODEL_PATH}")
        print("Run: bash scripts/download_vision_models.sh")
    
    print("\n" + "=" * 70)
    print("Summary of preprocessing steps for Rust implementation:")
    print("=" * 70)
    print("""
1. Convert to RGB (if not already)
2. Resize: shortest edge to 224, maintain aspect ratio (BICUBIC)
3. Center crop: 224x224
4. Convert to f32, rescale by 1/255 (pixel values 0-1)
5. Normalize: (pixel - mean) / std
   - mean = [0.48145466, 0.4578275, 0.40821073]
   - std = [0.26862954, 0.26130258, 0.27577711]
6. Reshape to NCHW: [1, 3, 224, 224]
7. Run ONNX inference
8. L2-normalize output (if not already normalized)
""")


if __name__ == "__main__":
    main()

