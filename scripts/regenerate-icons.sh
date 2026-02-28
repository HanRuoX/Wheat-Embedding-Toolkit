#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_IMAGE="${1:-$ROOT_DIR/public/wheat_logo.png}"
OUTPUT_DIR="$ROOT_DIR/src-tauri/icons"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/wheat-icons.XXXXXX")"
PREPARED_IMAGE="$WORK_DIR/icon-source.png"
ICON_FILL="${ICON_FILL:-940}"
ICON_STYLE="${ICON_STYLE:-rounded-square}"
ICON_BACKGROUND_COLOR="${ICON_BACKGROUND_COLOR:-#5B35A2}"
ICON_BACKGROUND_RADIUS="${ICON_BACKGROUND_RADIUS:-220}"

cleanup() {
  rm -rf "$WORK_DIR"
}

trap cleanup EXIT

if ! command -v magick >/dev/null 2>&1; then
  echo "Missing dependency: ImageMagick ('magick')."
  exit 1
fi

if ! command -v yarn >/dev/null 2>&1; then
  echo "Missing dependency: yarn."
  exit 1
fi

if [[ ! -f "$SOURCE_IMAGE" ]]; then
  echo "Source image not found: $SOURCE_IMAGE"
  exit 1
fi

echo "Preparing icon source from: $SOURCE_IMAGE"
if [[ "$ICON_STYLE" == "rounded-square" ]]; then
  FOREGROUND_IMAGE="$WORK_DIR/icon-foreground.png"
  BACKGROUND_IMAGE="$WORK_DIR/icon-background.png"

  magick "$SOURCE_IMAGE" \
    -trim +repage \
    -resize "${ICON_FILL}x${ICON_FILL}" \
    -gravity center \
    -background none \
    -extent 1024x1024 \
    "$FOREGROUND_IMAGE"

  magick -size 1024x1024 xc:none \
    -fill "$ICON_BACKGROUND_COLOR" \
    -draw "roundrectangle 0,0 1023,1023 ${ICON_BACKGROUND_RADIUS},${ICON_BACKGROUND_RADIUS}" \
    "$BACKGROUND_IMAGE"

  magick "$BACKGROUND_IMAGE" "$FOREGROUND_IMAGE" \
    -gravity center \
    -compose over \
    -composite \
    "$PREPARED_IMAGE"
else
  magick "$SOURCE_IMAGE" \
    -trim +repage \
    -resize "${ICON_FILL}x${ICON_FILL}" \
    -gravity center \
    -background none \
    -extent 1024x1024 \
    "$PREPARED_IMAGE"
fi

echo "Generating Tauri icons into: $OUTPUT_DIR"
(
  cd "$ROOT_DIR"
  yarn tauri icon "$PREPARED_IMAGE" -o "$OUTPUT_DIR"
)

echo "Done. Updated icons in $OUTPUT_DIR"
