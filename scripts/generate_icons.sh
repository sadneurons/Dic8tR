#!/bin/bash
# Generate PNG icons at standard hicolor sizes from SVG source
set -euo pipefail

SVG_SOURCE="${1:-assets/whispr.svg}"
OUTPUT_DIR="${2:-debian/icons}"

SIZES=(16 24 32 48 64 128 256)

mkdir -p "$OUTPUT_DIR"

for size in "${SIZES[@]}"; do
    rsvg-convert -w "$size" -h "$size" "$SVG_SOURCE" \
        -o "$OUTPUT_DIR/whispr-${size}.png"
    echo "Generated ${size}x${size}"
done

cp "$SVG_SOURCE" "$OUTPUT_DIR/whispr.svg"
echo "Done — icons in $OUTPUT_DIR"
