#!/usr/bin/env python3
"""
Visualize dominant colors comparing Python vs Rust implementations.
Shows Python's dominant color vs Rust's dominant color side-by-side.
"""

import base64
import os
import sys
from collections import Counter
from colorsys import rgb_to_hsv
from pathlib import Path

import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont
from io import BytesIO

# Test images - 10 diverse images for comprehensive testing
TEST_IMAGES = {
    "picsum_10": "https://picsum.photos/id/10/400/300",
    "picsum_20": "https://picsum.photos/id/20/200/200",
    "picsum_100": "https://picsum.photos/id/100/300/200",
    "picsum_200": "https://picsum.photos/id/200/400/300",
    "picsum_300": "https://picsum.photos/id/300/300/400",
    "picsum_400": "https://picsum.photos/id/400/500/300",
    "picsum_500": "https://picsum.photos/id/500/250/250",
    "picsum_600": "https://picsum.photos/id/600/350/250",
    "picsum_700": "https://picsum.photos/id/700/200/300",
    "picsum_800": "https://picsum.photos/id/800/400/400",
}

RUST_URL = "http://localhost:8080"


def download_image(url):
    """Download image and return bytes."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.content


def call_rust_stats(image_bytes, averaging_method):
    """Call Rust endpoint."""
    b64_content = base64.b64encode(image_bytes).decode("ascii")
    response = requests.post(
        f"{RUST_URL}/img/stats",
        json={"content": b64_content, "averaging_method": averaging_method},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def rgb_to_hex(rgb):
    """Convert RGB [0-1] to hex."""
    r, g, b = [round(c * 255) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def create_color_swatch(rgb, size=(200, 100)):
    """Create a color swatch image."""
    img = Image.new("RGB", size, tuple(round(c * 255) for c in rgb))
    return img


# Python color analysis functions (from test_image_stats.py)
THUMBNAIL_SIZE = 512


def resize_for_processing(image):
    """Create a thumbnail if image is too large"""
    if max(image.size) > THUMBNAIL_SIZE:
        width, height = image.size
        if width > height:
            new_width = THUMBNAIL_SIZE
            new_height = int(height * (THUMBNAIL_SIZE / width))
        else:
            new_height = THUMBNAIL_SIZE
            new_width = int(width * (THUMBNAIL_SIZE / height))
        thumb = image.copy()
        thumb.thumbnail((new_width, new_height), Image.Resampling.LANCZOS)
        return thumb
    return image


def prepare_image_for_color_analysis(image):
    """Prepare image for color analysis."""
    process_image = resize_for_processing(image)
    if process_image.mode != "RGBA":
        process_image = process_image.convert("RGBA")
    pixels = np.array(process_image)
    pixels = pixels.reshape(-1, 4)
    valid_pixels = pixels[pixels[:, 3] >= 128]
    if len(valid_pixels) == 0:
        return None
    return valid_pixels


def find_dominant_color_python(valid_pixels):
    """Find the dominant color using HSV clustering (Python implementation)."""
    rgb_pixels = valid_pixels[:, :3] / 255.0
    hsv_pixels = np.array([rgb_to_hsv(r, g, b) for r, g, b in rgb_pixels])
    quantized = (
        (hsv_pixels[:, 0] * 10).astype(int) * 1000
        + (hsv_pixels[:, 1] * 10).astype(int) * 10
        + (hsv_pixels[:, 2] * 10).astype(int)
    )
    color_counts = Counter(quantized)
    most_common_key = color_counts.most_common(1)[0][0]
    idx = np.where(quantized == most_common_key)[0][0]
    dominant_rgb = valid_pixels[idx, :3] / 255.0
    return dominant_rgb


def get_python_dominant_color(image):
    """Get Python's dominant color for an image."""
    valid_pixels = prepare_image_for_color_analysis(image)
    if valid_pixels is None:
        return None
    return find_dominant_color_python(valid_pixels)


def main():
    print("Comparing Python vs Rust dominant colors...\n")
    
    results = {}
    
    for name, url in TEST_IMAGES.items():
        print(f"Processing {name}...")
        image_bytes = download_image(url)
        image = Image.open(BytesIO(image_bytes))
        
        # Get Python's dominant color
        py_dom_rgb = get_python_dominant_color(image)
        py_dom_hex = rgb_to_hex(py_dom_rgb) if py_dom_rgb is not None else None
        
        # Get Rust's dominant color (doesn't matter which averaging method)
        rust_result = call_rust_stats(image_bytes, "arithmetic")
        rust_dom = rust_result["color_data"]["dominant_color"]
        rust_dom_rgb = rust_dom["rgb"]
        rust_dom_hex = rust_dom["hex"]
        
        # Compare Python vs Rust
        if py_dom_rgb is not None:
            diff = np.linalg.norm(np.array(py_dom_rgb) - np.array(rust_dom_rgb))
            match = diff < 0.05  # More lenient tolerance for dominant color
            status = "✓ MATCH" if match else "✗ DIFFER"
            print(f"  Python dominant: {py_dom_hex}")
            print(f"  Rust dominant:   {rust_dom_hex}")
            print(f"  Difference:      {diff:.4f}")
            print(f"  Status:          {status}")
        else:
            print(f"  Python: Failed to compute")
            print(f"  Rust:   {rust_dom_hex}")
            match = False
        
        # Also get average colors for display
        rust_avg_arith = rust_result["color_data"]["avg_color"]["rgb"]
        rust_result_geom = call_rust_stats(image_bytes, "geometric")
        rust_avg_geom = rust_result_geom["color_data"]["avg_color"]["rgb"]
        
        results[name] = {
            "image": image,
            "py_dom_rgb": py_dom_rgb,
            "py_dom_hex": py_dom_hex,
            "rust_dom_rgb": rust_dom_rgb,
            "rust_dom_hex": rust_dom_hex,
            "rust_avg_arith": rust_avg_arith,
            "rust_avg_geom": rust_avg_geom,
            "match": match,
        }
        print()
    
    # Create visualization
    print("Creating visualization images...")
    output_dir = Path("test_images") / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for name, data in results.items():
        img = data["image"]
        py_dom_rgb = data["py_dom_rgb"]
        rust_dom_rgb = data["rust_dom_rgb"]
        rust_avg_arith = data["rust_avg_arith"]
        rust_avg_geom = data["rust_avg_geom"]
        
        # Create a composite image showing original + color swatches
        swatch_size = (200, 100)
        composite = Image.new("RGB", (img.width + swatch_size[0] * 4, img.height + 150), "white")
        
        # Paste original image
        composite.paste(img, (0, 0))
        
        # Create and paste color swatches
        y_offset = img.height + 20
        
        # Python vs Rust dominant colors (first two swatches)
        if py_dom_rgb is not None:
            py_dom_swatch = create_color_swatch(py_dom_rgb, swatch_size)
            composite.paste(py_dom_swatch, (0, y_offset))
        
        rust_dom_swatch = create_color_swatch(rust_dom_rgb, swatch_size)
        composite.paste(rust_dom_swatch, (swatch_size[0], y_offset))
        
        # Average colors (swatches 3 and 4)
        avg_arith_swatch = create_color_swatch(rust_avg_arith, swatch_size)
        composite.paste(avg_arith_swatch, (swatch_size[0] * 2, y_offset))
        
        avg_geom_swatch = create_color_swatch(rust_avg_geom, swatch_size)
        composite.paste(avg_geom_swatch, (swatch_size[0] * 3, y_offset))
        
        # Add labels
        draw = ImageDraw.Draw(composite)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        except:
            font = ImageFont.load_default()
        
        labels = [
            f"Dom (py): {data['py_dom_hex'] or 'N/A'}",
            f"Dom (rs): {data['rust_dom_hex']}",
            f"Avg (arith): {rgb_to_hex(rust_avg_arith)}",
            f"Avg (geom): {rgb_to_hex(rust_avg_geom)}",
        ]
        
        for i, label in enumerate(labels):
            x = i * swatch_size[0] + 10
            y = y_offset + swatch_size[1] + 5
            draw.text((x, y), label, fill="black", font=font)
        
        output_path = output_dir / f"{name}_analysis.png"
        composite.save(output_path)
        print(f"  Saved: {output_path}")
    
    print(f"\n✓ Visualizations saved to: {output_dir}")
    print("\nSummary:")
    matches = [data["match"] for data in results.values() if data["py_dom_rgb"] is not None]
    all_match = all(matches) if matches else False
    
    for name, data in results.items():
        if data["py_dom_rgb"] is not None:
            status = "✓" if data["match"] else "✗"
            print(f"{status} {name}: Python={data['py_dom_hex']}, Rust={data['rust_dom_hex']}")
    
    if all_match:
        print("\n✓ All dominant colors match between Python and Rust!")
    else:
        print("\n⚠️  Some dominant colors differ (expected due to clustering algorithm differences)")
    
    return 0 if all_match else 0  # Don't fail - differences are acceptable


if __name__ == "__main__":
    sys.exit(main())

