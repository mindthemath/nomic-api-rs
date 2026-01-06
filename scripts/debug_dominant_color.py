#!/usr/bin/env python3
"""
Debug script to understand dominant color differences between Python and Rust.
"""

import base64
import sys
from collections import Counter
from colorsys import rgb_to_hsv
from io import BytesIO

import numpy as np
import requests
from PIL import Image

# Download the problematic image
url = "https://picsum.photos/id/10/400/300"
response = requests.get(url, timeout=30)
image_bytes = response.content
image = Image.open(BytesIO(image_bytes))

# Convert to RGBA and get valid pixels (same as Python reference)
if image.mode != "RGBA":
    image = image.convert("RGBA")
pixels = np.array(image)
pixels = pixels.reshape(-1, 4)
valid_pixels = pixels[pixels[:, 3] >= 128]

print(f"Total pixels: {len(pixels)}, Valid pixels: {len(valid_pixels)}")
print(f"Image size: {image.size}, mode: {image.mode}")

# Python implementation
rgb_pixels = valid_pixels[:, :3] / 255.0
hsv_pixels = np.array([rgb_to_hsv(r, g, b) for r, g, b in rgb_pixels])

# Quantize
quantized = (
    (hsv_pixels[:, 0] * 10).astype(int) * 1000
    + (hsv_pixels[:, 1] * 10).astype(int) * 10
    + (hsv_pixels[:, 2] * 10).astype(int)
)

color_counts = Counter(quantized)
most_common = color_counts.most_common(5)
print(f"\nPython - Top 5 quantized clusters:")
for key, count in most_common:
    idx = np.where(quantized == key)[0][0]
    rgb = valid_pixels[idx, :3] / 255.0
    h, s, v = hsv_pixels[idx]
    print(
        f"  Key={key:6d}, count={count:6d}, RGB=({rgb[0]:.3f}, {rgb[1]:.3f}, {rgb[2]:.3f}), HSV=({h:.3f}, {s:.3f}, {v:.3f})"
    )

# Get Python's dominant color
most_common_key = most_common[0][0]
idx = np.where(quantized == most_common_key)[0][0]
py_dominant_rgb = valid_pixels[idx, :3] / 255.0
py_hex = f"#{int(py_dominant_rgb[0]*255):02x}{int(py_dominant_rgb[1]*255):02x}{int(py_dominant_rgb[2]*255):02x}"
print(
    f"\nPython dominant color: RGB=({py_dominant_rgb[0]:.4f}, {py_dominant_rgb[1]:.4f}, {py_dominant_rgb[2]:.4f}), hex={py_hex}"
)

# Call Rust endpoint
b64_content = base64.b64encode(image_bytes).decode("ascii")
rust_result = requests.post(
    "http://localhost:8080/img/stats",
    json={"content": b64_content, "averaging_method": "arithmetic"},
    headers={"Content-Type": "application/json"},
    timeout=60,
).json()

rust_dominant = rust_result["color_data"]["dominant_color"]
print(
    f"\nRust dominant color: RGB=({rust_dominant['rgb'][0]:.4f}, {rust_dominant['rgb'][1]:.4f}, {rust_dominant['rgb'][2]:.4f}), hex={rust_dominant['hex']}"
)

# Check if the Rust color exists in our valid pixels
rust_rgb = np.array(rust_dominant["rgb"])
print(f"\nSearching for Rust color in valid pixels...")
min_dist = float("inf")
closest_idx = -1
for i, pixel_rgb in enumerate(rgb_pixels):
    dist = np.linalg.norm(pixel_rgb - rust_rgb)
    if dist < min_dist:
        min_dist = dist
        closest_idx = i

if closest_idx >= 0:
    closest_rgb = rgb_pixels[closest_idx]
    h, s, v = hsv_pixels[closest_idx]
    q_key = quantized[closest_idx]
    print(
        f"Closest pixel: RGB=({closest_rgb[0]:.4f}, {closest_rgb[1]:.4f}, {closest_rgb[2]:.4f}), HSV=({h:.3f}, {s:.3f}, {v:.3f}), quantized_key={q_key}"
    )
    print(f"Distance from Rust color: {min_dist:.6f}")

    # Check if this key is in top clusters
    for rank, (key, count) in enumerate(most_common):
        if key == q_key:
            print(f"This key is ranked #{rank+1} with count {count}")
