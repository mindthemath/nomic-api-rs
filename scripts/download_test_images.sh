#!/bin/bash
# Download test images used in validation for local inspection
# Uses the same deterministic generation as test_image_stats.py

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_IMAGES_DIR="$PROJECT_ROOT/test_images"

mkdir -p "$TEST_IMAGES_DIR"

# Default seed and count (match test_image_stats.py defaults)
SEED=${1:-42}
COUNT=${2:-10}

echo "Downloading test images to $TEST_IMAGES_DIR..."
echo "Using seed=$SEED, count=$COUNT"
echo ""

# Generate URLs using Python (same logic as test_image_stats.py)
python3 << EOF
import sys
sys.path.insert(0, '$SCRIPT_DIR')
from test_image_stats import generate_test_images

urls = generate_test_images(seed=$SEED, count=$COUNT)
for url in urls:
    # Extract ID and dimensions from URL
    # Format: https://picsum.photos/id/{id}/{width}/{height}
    parts = url.split('/')
    img_id = parts[-3]
    width = parts[-2]
    height = parts[-1]
    filename = f"picsum_{img_id}_{width}x{height}.jpg"
    print(f"{url}|{filename}")
EOF | while IFS='|' read -r url filename; do
    echo "  Downloading $filename..."
    curl -s -L -o "$TEST_IMAGES_DIR/$filename" "$url" || echo "    ⚠️  Failed to download $filename"
done

echo ""
echo "✓ Images downloaded:"
ls -lh "$TEST_IMAGES_DIR"/*.jpg 2>/dev/null | wc -l | xargs echo "  Total files:"

echo ""
echo "Images saved to: $TEST_IMAGES_DIR"
echo "You can now visually inspect them to validate the color analysis results."
echo ""
echo "Usage: $0 [seed] [count]"
echo "  seed: Random seed for deterministic selection (default: 42)"
echo "  count: Number of images to download (default: 10)"

