#!/bin/bash
# Download test images used in validation for local inspection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TEST_IMAGES_DIR="$PROJECT_ROOT/test_images"

mkdir -p "$TEST_IMAGES_DIR"

echo "Downloading test images to $TEST_IMAGES_DIR..."

# Download the test images used in test_image_stats.py (10 total)
curl -s -L -o "$TEST_IMAGES_DIR/picsum_10_400x300.jpg" "https://picsum.photos/id/10/400/300"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_20_200x200.jpg" "https://picsum.photos/id/20/200/200"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_100_300x200.jpg" "https://picsum.photos/id/100/300/200"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_200_400x300.jpg" "https://picsum.photos/id/200/400/300"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_300_300x400.jpg" "https://picsum.photos/id/300/300/400"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_400_500x300.jpg" "https://picsum.photos/id/400/500/300"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_500_250x250.jpg" "https://picsum.photos/id/500/250/250"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_600_350x250.jpg" "https://picsum.photos/id/600/350/250"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_700_200x300.jpg" "https://picsum.photos/id/700/200/300"
curl -s -L -o "$TEST_IMAGES_DIR/picsum_800_400x400.jpg" "https://picsum.photos/id/800/400/400"

echo "✓ Images downloaded:"
ls -lh "$TEST_IMAGES_DIR"/*.jpg

echo ""
echo "Images saved to: $TEST_IMAGES_DIR"
echo "You can now visually inspect them to validate the color analysis results."

