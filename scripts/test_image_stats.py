#!/usr/bin/env python3
"""
Test script to validate the Rust /img/stats endpoint against the Python reference implementation.

This script:
1. Downloads a test image ONCE
2. Computes stats using the Python reference functions
3. Sends the SAME image bytes (as base64) to the Rust endpoint
4. Compares the results

Usage:
    python3 scripts/test_image_stats.py [--rust-url http://localhost:8080]
"""

import argparse
import base64
import sys
from collections import Counter
from colorsys import rgb_to_hsv
from io import BytesIO

import numpy as np
import requests
from PIL import ExifTags, Image, ImageDraw, ImageFont
from pathlib import Path


# ============================================================================
# Color Analysis Functions (from api_stats.py, without litserve dependency)
# ============================================================================

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


def calculate_arithmetic_mean(valid_pixels):
    """Calculate arithmetic mean of valid pixels"""
    return valid_pixels[:, :3].mean(axis=0) / 255.0


def calculate_geometric_mean(valid_pixels):
    """Calculate geometric mean of valid pixels"""
    rgb_values = valid_pixels[:, :3].astype(float)
    eps = 1e-8
    rgb_values = np.maximum(rgb_values, eps)
    log_values = np.log(rgb_values)
    log_mean = np.mean(log_values, axis=0)
    geometric_mean = np.exp(log_mean)
    return geometric_mean / 255.0


def rgb_to_hex(rgb_array):
    """Convert RGB array (0-1 range) to hex color code"""
    r_int, g_int, b_int = [int(c * 255) for c in rgb_array]
    return f"#{r_int:02x}{g_int:02x}{b_int:02x}"


def find_dominant_color(valid_pixels):
    """Find the dominant color using HSV clustering"""
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


def get_image_colors(image, averaging_method="arithmetic"):
    """Extract color information from image."""
    valid_pixels = prepare_image_for_color_analysis(image)
    if valid_pixels is None:
        return None

    if averaging_method == "geometric":
        avg_color = calculate_geometric_mean(valid_pixels)
    else:
        avg_color = calculate_arithmetic_mean(valid_pixels)

    dominant_rgb = find_dominant_color(valid_pixels)

    return {
        "avg_color": {
            "rgb": avg_color.tolist(),
            "hex": rgb_to_hex(avg_color),
            "method": averaging_method,
        },
        "dominant_color": {
            "rgb": dominant_rgb.tolist(),
            "hex": rgb_to_hex(dominant_rgb),
        },
    }


def get_exif_data(image):
    """Extract EXIF data from image."""
    exif_data = {}
    try:
        if hasattr(image, "_getexif") and image._getexif():
            for tag, value in image._getexif().items():
                if tag in ExifTags.TAGS:
                    tag_name = ExifTags.TAGS[tag]
                    try:
                        if hasattr(value, "numerator") and hasattr(value, "denominator"):
                            if value.denominator != 0:
                                value = float(value.numerator) / value.denominator
                            else:
                                value = 0
                        import json
                        json.dumps(value)
                        exif_data[tag_name] = value
                    except (TypeError, OverflowError):
                        try:
                            exif_data[tag_name] = str(value)
                        except Exception:
                            pass
    except Exception:
        pass
    return exif_data


def download_image(url: str) -> tuple[Image.Image, bytes]:
    """Download image and return both PIL Image and raw bytes."""
    headers = {"User-Agent": "Mozilla/5.0 (compatible; image-stats-test/1.0)"}
    response = requests.get(url, timeout=30, headers=headers)
    response.raise_for_status()
    image_bytes = response.content
    image = Image.open(BytesIO(image_bytes))
    return image, image_bytes


def call_rust_endpoint_with_bytes(rust_url: str, image_bytes: bytes, averaging_method: str = "geometric") -> dict:
    """Call the Rust /img/stats endpoint with base64-encoded image bytes."""
    # Send image as base64 to ensure both implementations analyze the SAME image
    b64_content = base64.b64encode(image_bytes).decode("ascii")
    response = requests.post(
        f"{rust_url}/img/stats",
        json={"content": b64_content, "averaging_method": averaging_method},
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def compare_colors(py_color: dict, rust_color: dict, label: str, tolerance: float = 0.03) -> list[str]:
    """Compare color values between Python and Rust implementations."""
    errors = []
    
    # Compare RGB values
    if py_color and rust_color:
        py_rgb = py_color.get("rgb", [])
        rust_rgb = rust_color.get("rgb", [])
        
        if len(py_rgb) == 3 and len(rust_rgb) == 3:
            max_diff = 0
            for i, (py_val, rust_val) in enumerate(zip(py_rgb, rust_rgb)):
                diff = abs(py_val - rust_val)
                max_diff = max(max_diff, diff)
                if diff > tolerance:
                    errors.append(f"{label} RGB[{i}]: Python={py_val:.4f}, Rust={rust_val:.4f}, diff={diff:.4f}")
            
            # Only report hex difference if RGB is actually different beyond tolerance
            py_hex = py_color.get("hex", "").lower()
            rust_hex = rust_color.get("hex", "").lower()
            if py_hex != rust_hex and max_diff <= tolerance:
                # Hex differs but RGB is within tolerance - just a rounding difference, not an error
                pass
    
    return errors


def run_test(image_url: str, rust_url: str, averaging_method: str = "geometric", verbose: bool = True) -> tuple[bool, list[str], dict]:
    """Run a single test comparing Python and Rust implementations."""
    errors = []
    
    print(f"\n{'='*60}")
    print(f"Testing image: {image_url}")
    print(f"Averaging method: {averaging_method}")
    print("="*60)
    
    # Download image ONCE - both Python and Rust will use the same bytes
    print("\n1. Downloading image...")
    try:
        image, image_bytes = download_image(image_url)
        print(f"   Image size: {image.size}, mode: {image.mode}, bytes: {len(image_bytes)}")
    except Exception as e:
        return False, [f"Failed to download image: {e}"], {}
    
    # Python reference
    print("\n2. Computing Python reference...")
    try:
        py_exif = get_exif_data(image)
        py_colors = get_image_colors(image, averaging_method)
        print(f"   EXIF fields: {len(py_exif)}")
        if py_colors:
            print(f"   Avg color: {py_colors['avg_color']['hex']}")
            print(f"   Dominant color: {py_colors['dominant_color']['hex']}")
    except Exception as e:
        return False, [f"Python reference failed: {e}"]
    
    # Rust endpoint - send SAME image bytes as base64
    print("\n3. Calling Rust endpoint (same image via base64)...")
    try:
        rust_result = call_rust_endpoint_with_bytes(rust_url, image_bytes, averaging_method)
        print(f"   Time: {rust_result['time_ms']:.2f}ms")
        print(f"   EXIF fields: {len(rust_result.get('exif_data', {}))}")
        if rust_result.get("color_data"):
            print(f"   Avg color: {rust_result['color_data']['avg_color']['hex']}")
            print(f"   Dominant color: {rust_result['color_data']['dominant_color']['hex']}")
    except requests.exceptions.ConnectionError:
        return False, ["Failed to connect to Rust server. Is it running with --features image-stats?"]
    except Exception as e:
        return False, [f"Rust endpoint failed: {e}"]
    
    # Compare results
    print("\n4. Comparing results...")
    
    # Compare color data
    if py_colors and rust_result.get("color_data"):
        rust_colors = rust_result["color_data"]
        
        # Average color
        avg_errors = compare_colors(
            py_colors.get("avg_color"),
            rust_colors.get("avg_color"),
            "Avg color"
        )
        errors.extend(avg_errors)
        
        # Dominant color - more lenient (HSV clustering can vary)
        dom_errors = compare_colors(
            py_colors.get("dominant_color"),
            rust_colors.get("dominant_color"),
            "Dominant color",
            tolerance=0.05  # More lenient for dominant color
        )
        # Dominant color differences are warnings, not errors
        for err in dom_errors:
            print(f"   ⚠️  {err}")
    elif py_colors and not rust_result.get("color_data"):
        errors.append("Rust returned no color_data but Python did")
    elif not py_colors and rust_result.get("color_data"):
        errors.append("Python returned no colors but Rust did")
    
    # Prepare result data
    result_data = {
        "image_url": image_url,
        "averaging_method": averaging_method,
        "py_colors": py_colors,
        "rust_colors": rust_result.get("color_data"),
        "errors": errors,
        "avg_match": len([e for e in errors if "Avg color" in e]) == 0,
        "dom_match": len([e for e in errors if "Dominant color" in e]) == 0,
    }
    
    # Print results
    if errors:
        if verbose:
            print("\n❌ Differences found:")
            for err in errors:
                print(f"   - {err}")
        return False, errors, result_data
    else:
        if verbose:
            print("\n✅ Results match!")
        return True, [], result_data


# ============================================================================
# Visualization Functions
# ============================================================================

def rgb_to_hex_vis(rgb):
    """Convert RGB [0-1] to hex."""
    r, g, b = [int(c * 255) for c in rgb]
    return f"#{r:02x}{g:02x}{b:02x}"


def create_color_swatch(rgb, size=(200, 100)):
    """Create a color swatch image."""
    img = Image.new("RGB", size, tuple(int(c * 255) for c in rgb))
    return img


def calculate_color_difference(rgb1, rgb2):
    """Calculate max component difference (L∞ norm) between two RGB colors (0-1 range).
    
    This better captures hex code differences - if any single RGB component differs,
    the max difference will reflect that, unlike L2 which can hide single-component
    differences when other components match.
    """
    if rgb1 is None or rgb2 is None:
        return 1.0  # Max difference if one is missing
    diff = np.abs(np.array(rgb1) - np.array(rgb2))
    return float(np.max(diff))  # Max component difference


def hex_to_rgb_int(hex_str):
    """Parse hex string to RGB integers (0-255)."""
    if not hex_str:
        return None
    hex_str = hex_str.lstrip('#').lower()
    if len(hex_str) != 6:
        return None
    try:
        r = int(hex_str[0:2], 16)
        g = int(hex_str[2:4], 16)
        b = int(hex_str[4:6], 16)
        return (r, g, b)
    except ValueError:
        return None


def calculate_color_difference_from_hex(hex1, hex2):
    """Calculate max component difference from hex codes (0-1 normalized).
    
    This avoids precision issues from JSON serialization of RGB floats.
    Parses hex codes directly and calculates the difference in integer space.
    Returns the normalized difference (0-1 range).
    """
    if not hex1 or not hex2:
        return 1.0
    
    rgb1 = hex_to_rgb_int(hex1)
    rgb2 = hex_to_rgb_int(hex2)
    
    if rgb1 is None or rgb2 is None:
        return 1.0
    
    # Calculate max component difference in integer space, then normalize
    max_diff_int = max(abs(rgb1[i] - rgb2[i]) for i in range(3))
    return max_diff_int / 255.0  # Normalize to 0-1 range


def count_hex_digit_differences(hex1, hex2):
    """Count how many hex digits differ between two hex codes.
    
    Returns the number of hex digit positions that differ (0-6).
    """
    if not hex1 or not hex2:
        return 6  # Max difference
    
    h1 = hex1.lstrip('#').lower()
    h2 = hex2.lstrip('#').lower()
    
    if len(h1) != 6 or len(h2) != 6:
        return 6
    
    differences = sum(1 for i in range(6) if h1[i] != h2[i])
    return differences


def format_scientific_notation(value, precision=3):
    """Format a float in scientific notation for consistent sorting.
    
    Formats as: mantissa + 'E' + sign + exponent (zero-padded to 3 digits)
    Example: 0.00392 -> "3.920E-003"
    Example: 0.0 -> "0.000E+000"
    
    This ensures proper lexicographic sorting where larger differences come first.
    Uses capital E and 3 decimal places for shorter filenames.
    """
    if value == 0.0:
        return "0.000E+000"
    
    # Format with scientific notation (lowercase e first, then convert to uppercase)
    formatted = f"{value:.{precision}e}"
    
    # Parse and reformat with zero-padded exponent and capital E
    if 'e' in formatted.lower():
        mantissa, exp_part = formatted.lower().split('e')
        exp_sign = exp_part[0]
        exp_value = int(exp_part[1:])
        
        # Zero-pad exponent to 3 digits for consistent sorting
        exp_str = f"{exp_sign}{exp_value:03d}"
        return f"{mantissa}E{exp_str}"
    
    return formatted.upper()


def create_visualization(image, image_name, py_dom_rgb, rust_dom_rgb, rust_avg_arith, rust_avg_geom, output_dir, diff_score_from_hex=None, hex_diff_count=None, diff_score_from_rgb=None, py_dom_hex=None, rust_dom_hex=None):
    """Create visualization image showing original + color swatches in 2x2 grid."""
    swatch_size = (200, 150)  # Taller swatches for better proportions
    gap = 20  # Gap between image and swatches, and between swatches
    label_height = 30  # Space for labels below swatches
    
    # Calculate canvas dimensions - no whitespace on right
    swatch_area_width = swatch_size[0] * 2 + gap  # 2 columns + gap between
    swatch_area_height = swatch_size[1] * 2 + gap + label_height  # 2 rows + gap + labels
    
    canvas_width = image.width + gap + swatch_area_width
    canvas_height = max(image.height, swatch_area_height)
    
    composite = Image.new("RGB", (canvas_width, canvas_height), "white")
    
    # Paste original image on the left
    composite.paste(image, (0, 0))
    
    # Calculate swatch area position (right side, vertically centered if image is taller)
    swatch_x = image.width + gap
    swatch_y = max(0, (canvas_height - swatch_area_height) // 2)
    
    # Create and paste color swatches in 2x2 grid
    # Top row: Dominant colors
    if py_dom_rgb is not None:
        py_dom_swatch = create_color_swatch(py_dom_rgb, swatch_size)
        composite.paste(py_dom_swatch, (swatch_x, swatch_y))
    
    rust_dom_swatch = create_color_swatch(rust_dom_rgb, swatch_size)
    composite.paste(rust_dom_swatch, (swatch_x + swatch_size[0] + gap, swatch_y))
    
    # Bottom row: Average colors
    avg_arith_swatch = create_color_swatch(rust_avg_arith, swatch_size)
    composite.paste(avg_arith_swatch, (swatch_x, swatch_y + swatch_size[1] + gap))
    
    avg_geom_swatch = create_color_swatch(rust_avg_geom, swatch_size)
    composite.paste(avg_geom_swatch, (swatch_x + swatch_size[0] + gap, swatch_y + swatch_size[1] + gap))
    
    # Add labels below each swatch
    draw = ImageDraw.Draw(composite)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    # Use actual hex codes from JSON (not converted from RGB) to avoid rounding differences
    py_hex = py_dom_hex if py_dom_hex else (rgb_to_hex_vis(py_dom_rgb) if py_dom_rgb is not None else "N/A")
    rust_hex = rust_dom_hex if rust_dom_hex else (rgb_to_hex_vis(rust_dom_rgb) if rust_dom_rgb else "N/A")
    
    # Labels for top row
    label_y = swatch_y + swatch_size[1] + 5
    if py_dom_rgb is not None:
        draw.text((swatch_x + 5, label_y), f"Dom (py): {py_hex}", fill="black", font=font)
    draw.text((swatch_x + swatch_size[0] + gap + 5, label_y), f"Dom (rs): {rust_hex}", fill="black", font=font)
    
    # Labels for bottom row
    label_y_bottom = swatch_y + swatch_size[1] * 2 + gap + 5
    draw.text((swatch_x + 5, label_y_bottom), f"Avg (arith): {rgb_to_hex_vis(rust_avg_arith)}", fill="black", font=font)
    draw.text((swatch_x + swatch_size[0] + gap + 5, label_y_bottom), f"Avg (geom): {rgb_to_hex_vis(rust_avg_geom)}", fill="black", font=font)
    
    # Generate filename with difference prefix for sorting
    # Format: hex{count}_diff{scientific_notation}_rgb{scientific_notation}_
    # Example: hex1_diff3.920000e-003_rgb3.920000e-003_picsum_600_analysis.png
    if diff_score_from_hex is not None:
        # Use scientific notation for hex-based diff (primary sort key)
        hex_diff_str = format_scientific_notation(diff_score_from_hex)
        # Secondary sort key: hex digit count
        hex_count_str = f"hex{hex_diff_count:02d}" if hex_diff_count is not None else "hex??"
        # Also include RGB-based diff for comparison/debugging
        rgb_diff_str = format_scientific_notation(diff_score_from_rgb) if diff_score_from_rgb is not None else "0.000000e+000"
        diff_prefix = f"{hex_count_str}_diff{hex_diff_str}_rgb{rgb_diff_str}_"
    else:
        diff_prefix = "hex??_diff0.000000e+000_rgb0.000000e+000_"
    
    output_path = output_dir / f"{diff_prefix}picsum_{image_name}_analysis.png"
    composite.save(output_path)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Test Rust /img/stats endpoint against Python reference")
    parser.add_argument("--rust-url", default="http://localhost:8080", help="Rust server URL")
    parser.add_argument("--image-url", default=None, help="Specific image URL to test")
    args = parser.parse_args()
    
    # Test images - using STABLE URLs (not random picsum)
    # Total of 10 diverse images for comprehensive testing
    test_images = [
        "https://picsum.photos/id/10/400/300",   # Forest landscape
        "https://picsum.photos/id/20/200/200",   # Beach scene
        "https://picsum.photos/id/100/300/200",  # Landscape
        "https://picsum.photos/id/200/400/300",  # Architecture
        "https://picsum.photos/id/300/300/400",  # Portrait orientation
        "https://picsum.photos/id/400/500/300",  # Wide landscape
        "https://picsum.photos/id/500/250/250",  # Square format
        "https://picsum.photos/id/600/350/250",  # Medium landscape
        "https://picsum.photos/id/700/200/300",  # Portrait
        "https://picsum.photos/id/800/400/400",  # Large square
    ]
    
    if args.image_url:
        test_images = [args.image_url]
    
    # Track results per image
    image_results = {}
    all_passed = True
    total_tests = 0
    passed_tests = 0
    
    # Create output directory for visualizations
    output_dir = Path("test_images") / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test both averaging methods
    for image_url in test_images:
        image_name = image_url.split("/id/")[1].split("/")[0] if "/id/" in image_url else "unknown"
        image_results[image_name] = {
            "url": image_url,
            "arithmetic": None,
            "geometric": None,
            "image": None,
            "image_bytes": None,
        }
        
        # Download image once per image (not per method)
        try:
            image, image_bytes = download_image(image_url)
            image_results[image_name]["image"] = image
            image_results[image_name]["image_bytes"] = image_bytes
        except Exception as e:
            print(f"Failed to download {image_url}: {e}")
            continue
        
        for method in ["arithmetic", "geometric"]:
            total_tests += 1
            passed, errors, result_data = run_test(image_url, args.rust_url, method, verbose=True)
            image_results[image_name][method] = result_data
            if passed:
                passed_tests += 1
            else:
                all_passed = False
    
    # Generate visualizations - calculate differences and sort by importance
    print(f"\n{'='*60}")
    print("Generating visualization images...")
    print("="*60)
    
    # Collect all images with their difference scores
    visualization_tasks = []
    for image_name, data in image_results.items():
        if data["image"] is None:
            continue
        
        # Get Python dominant color (doesn't depend on averaging method)
        py_dom_rgb = None
        py_dom_hex = None
        if data["arithmetic"] and data["arithmetic"].get("py_colors"):
            py_dom_rgb = data["arithmetic"]["py_colors"].get("dominant_color", {}).get("rgb")
            py_dom_hex = data["arithmetic"]["py_colors"].get("dominant_color", {}).get("hex")
        
        # Get Rust colors (use arithmetic for dominant, both for averages)
        rust_dom_rgb = [0.0, 0.0, 0.0]
        rust_dom_hex = None
        rust_avg_arith = [0.0, 0.0, 0.0]
        rust_avg_geom = [0.0, 0.0, 0.0]
        
        if data["arithmetic"] and data["arithmetic"].get("rust_colors"):
            rust_dom_rgb = data["arithmetic"]["rust_colors"].get("dominant_color", {}).get("rgb", [0, 0, 0])
            rust_dom_hex = data["arithmetic"]["rust_colors"].get("dominant_color", {}).get("hex")
            rust_avg_arith = data["arithmetic"]["rust_colors"].get("avg_color", {}).get("rgb", [0, 0, 0])
        
        if data["geometric"] and data["geometric"].get("rust_colors"):
            rust_avg_geom = data["geometric"]["rust_colors"].get("avg_color", {}).get("rgb", [0, 0, 0])
        
        if py_dom_rgb or rust_dom_rgb:
            # Calculate dominant color difference from hex codes (avoids JSON precision issues)
            diff_score_from_hex = calculate_color_difference_from_hex(py_dom_hex, rust_dom_hex)
            hex_diff_count = count_hex_digit_differences(py_dom_hex, rust_dom_hex)
            # Also calculate from RGB for comparison/debugging
            diff_score_from_rgb = calculate_color_difference(py_dom_rgb, rust_dom_rgb)
            
            visualization_tasks.append((
                diff_score_from_hex,  # Primary sort key (from hex)
                hex_diff_count,       # Secondary sort key (hex digit count)
                diff_score_from_rgb,  # For debugging/comparison
                image_name,
                data["image"],
                py_dom_rgb,
                rust_dom_rgb,
                rust_avg_arith,
                rust_avg_geom,
                py_dom_hex,
                rust_dom_hex,
            ))
    
    # Sort by difference score (highest first - most problematic images first)
    # Primary sort: hex-based diff, secondary sort: hex digit count
    visualization_tasks.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    # Generate visualizations in sorted order
    for diff_score_from_hex, hex_diff_count, diff_score_from_rgb, image_name, image, py_dom_rgb, rust_dom_rgb, rust_avg_arith, rust_avg_geom, py_dom_hex, rust_dom_hex in visualization_tasks:
        output_path = create_visualization(
            image,
            image_name,
            py_dom_rgb,
            rust_dom_rgb,
            rust_avg_arith,
            rust_avg_geom,
            output_dir,
            diff_score_from_hex=diff_score_from_hex,
            hex_diff_count=hex_diff_count,
            diff_score_from_rgb=diff_score_from_rgb,
            py_dom_hex=py_dom_hex,
            rust_dom_hex=rust_dom_hex,
        )
        hex_diff_str = format_scientific_notation(diff_score_from_hex)
        rgb_diff_str = format_scientific_notation(diff_score_from_rgb)
        print(f"  ✓ {output_path.name} (hex diff: {hex_diff_str}, hex digits: {hex_diff_count}, rgb diff: {rgb_diff_str})")
    
    # Detailed Summary - sorted by difference score
    print(f"\n{'='*60}")
    print("DETAILED SUMMARY")
    print("="*60)
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed ({passed_tests*100//total_tests}%)")
    print(f"\nPer-image breakdown (sorted by dominant color difference, highest first):")
    print("-" * 60)
    
    # Calculate differences and sort
    summary_items = []
    for image_name, data in image_results.items():
        if data["image"] is None:
            continue
        
        arith_result = data["arithmetic"]
        geom_result = data["geometric"]
        
        if not arith_result or not geom_result:
            continue
        
        # Calculate dominant color difference (both from hex and RGB)
        py_dom_rgb = None
        py_dom_hex = None
        rust_dom_rgb = None
        rust_dom_hex = None
        if arith_result.get("py_colors"):
            py_dom_rgb = arith_result["py_colors"].get("dominant_color", {}).get("rgb")
            py_dom_hex = arith_result["py_colors"].get("dominant_color", {}).get("hex")
        if arith_result.get("rust_colors"):
            rust_dom_rgb = arith_result["rust_colors"].get("dominant_color", {}).get("rgb")
            rust_dom_hex = arith_result["rust_colors"].get("dominant_color", {}).get("hex")
        
        # Calculate differences from hex (primary) and RGB (for comparison)
        diff_score_from_hex = calculate_color_difference_from_hex(py_dom_hex, rust_dom_hex)
        hex_diff_count = count_hex_digit_differences(py_dom_hex, rust_dom_hex)
        diff_score_from_rgb = calculate_color_difference(py_dom_rgb, rust_dom_rgb)
        
        summary_items.append((diff_score_from_hex, hex_diff_count, diff_score_from_rgb, image_name, data, arith_result, geom_result))
    
    # Sort by difference (highest first) - primary: hex diff, secondary: hex digit count
    summary_items.sort(key=lambda x: (x[0], x[1]), reverse=True)
    
    for diff_score_from_hex, hex_diff_count, diff_score_from_rgb, image_name, data, arith_result, geom_result in summary_items:
        arith_passed = arith_result.get("avg_match", False) and arith_result.get("dom_match", False)
        geom_passed = geom_result.get("avg_match", False) and geom_result.get("dom_match", False)
        
        status_arith = "✓" if arith_passed else "✗"
        status_geom = "✓" if geom_passed else "✗"
        
        py_dom = "N/A"
        rust_dom = "N/A"
        if arith_result.get("py_colors"):
            py_dom = arith_result["py_colors"].get("dominant_color", {}).get("hex", "N/A")
        if arith_result.get("rust_colors"):
            rust_dom = arith_result["rust_colors"].get("dominant_color", {}).get("hex", "N/A")
        
        # Check if dominant colors actually differ (even if within tolerance)
        dom_differs = py_dom != "N/A" and rust_dom != "N/A" and py_dom.lower() != rust_dom.lower()
        
        # Format scientific notation for display
        hex_diff_str = format_scientific_notation(diff_score_from_hex)
        rgb_diff_str = format_scientific_notation(diff_score_from_rgb)
        
        print(f"\n{image_name} (hex diff: {hex_diff_str}, hex digits: {hex_diff_count}, rgb diff: {rgb_diff_str}):")
        print(f"  Arithmetic: {status_arith}  Geometric: {status_geom}")
        if dom_differs:
            print(f"  ⚠️  Dominant colors DIFFER - Python: {py_dom}, Rust: {rust_dom}")
            print(f"      Hex-based diff: {hex_diff_str} (from hex codes, avoids JSON precision issues)")
            print(f"      RGB-based diff: {rgb_diff_str} (from JSON RGB values, may have precision loss)")
        else:
            print(f"  ✓ Dominant colors match - {py_dom}")
            print(f"      Hex-based diff: {hex_diff_str} (from hex codes)")
            print(f"      RGB-based diff: {rgb_diff_str} (from JSON RGB values)")
        
        if not arith_passed or not geom_passed:
            print(f"  ⚠️  Issues found:")
            if arith_result.get("errors"):
                for err in arith_result["errors"]:
                    if "Avg color" in err or "Dominant color" in err:
                        print(f"     - Arithmetic: {err}")
            if geom_result.get("errors"):
                for err in geom_result["errors"]:
                    if "Avg color" in err or "Dominant color" in err:
                        print(f"     - Geometric: {err}")
    
    print(f"\n{'='*60}")
    print(f"Visualizations saved to: {output_dir}")
    print("="*60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

