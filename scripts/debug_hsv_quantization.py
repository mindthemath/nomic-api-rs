#!/usr/bin/env python3
"""
Compare HSV quantization between Python and Rust implementations.
"""

import base64
import requests
from colorsys import rgb_to_hsv
from PIL import Image
from io import BytesIO
import numpy as np

# Download image
url = "https://picsum.photos/id/10/400/300"
response = requests.get(url, timeout=30)
image_bytes = response.content
image = Image.open(BytesIO(image_bytes))

# Convert to RGBA
if image.mode != "RGBA":
    image = image.convert("RGBA")
pixels = np.array(image)
pixels = pixels.reshape(-1, 4)
valid_pixels = pixels[pixels[:, 3] >= 128]

print(f"Valid pixels: {len(valid_pixels)}")

# Sample first 10 pixels and compare quantization
print("\nFirst 10 pixels - Python quantization:")
for i in range(min(10, len(valid_pixels))):
    r, g, b = valid_pixels[i, :3] / 255.0
    h, s, v = rgb_to_hsv(r, g, b)
    h_q = int(h * 10)
    s_q = int(s * 10)
    v_q = int(v * 10)
    key = h_q * 1000 + s_q * 10 + v_q
    print(f"  Pixel {i}: RGB=({r:.3f}, {g:.3f}, {b:.3f}), HSV=({h:.3f}, {s:.3f}, {v:.3f}), quantized=({h_q}, {s_q}, {v_q}), key={key}")

# Count all quantized keys
rgb_pixels = valid_pixels[:, :3] / 255.0
hsv_pixels = np.array([rgb_to_hsv(r, g, b) for r, g, b in rgb_pixels])
quantized = (
    (hsv_pixels[:, 0] * 10).astype(int) * 1000
    + (hsv_pixels[:, 1] * 10).astype(int) * 10
    + (hsv_pixels[:, 2] * 10).astype(int)
)

from collections import Counter
counts = Counter(quantized)
top5 = counts.most_common(5)

print("\nTop 5 clusters (Python):")
for key, count in top5:
    # Find a pixel with this key
    idx = np.where(quantized == key)[0][0]
    r, g, b = rgb_pixels[idx]
    h, s, v = hsv_pixels[idx]
    h_q = int(h * 10)
    s_q = int(s * 10)
    v_q = int(v * 10)
    print(f"  Key={key:6d}, count={count:6d}, HSV=({h:.3f}, {s:.3f}, {v:.3f}), quantized=({h_q}, {s_q}, {v_q})")

# Now call Rust and see what it reports
print("\nCalling Rust endpoint...")
b64_content = base64.b64encode(image_bytes).decode("ascii")
rust_result = requests.post(
    "http://localhost:8080/img/stats",
    json={"content": b64_content, "averaging_method": "arithmetic"},
    headers={"Content-Type": "application/json"},
    timeout=60,
).json()

rust_dom = rust_result["color_data"]["dominant_color"]["rgb"]
print(f"Rust dominant: RGB=({rust_dom[0]:.4f}, {rust_dom[1]:.4f}, {rust_dom[2]:.4f})")

# Find which cluster this belongs to in Python
rust_h, rust_s, rust_v = rgb_to_hsv(rust_dom[0], rust_dom[1], rust_dom[2])
rust_h_q = int(rust_h * 10)
rust_s_q = int(rust_s * 10)
rust_v_q = int(rust_v * 10)
rust_key = rust_h_q * 1000 + rust_s_q * 10 + rust_v_q
print(f"Rust HSV: ({rust_h:.3f}, {rust_s:.3f}, {rust_v:.3f}), quantized=({rust_h_q}, {rust_s_q}, {rust_v_q}), key={rust_key}")

# Check if this key exists in our counts
if rust_key in counts:
    print(f"This key has count {counts[rust_key]} in Python (rank: {sorted(counts.values(), reverse=True).index(counts[rust_key]) + 1})")
else:
    print("This key does NOT exist in Python's quantization!")

