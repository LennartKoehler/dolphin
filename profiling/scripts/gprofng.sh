#!/bin/bash


# Set the directory you want to run the command from
TARGET_DIR="debug_build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTNAME="../results/gprofng/test.er/"
# Change to that directory
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
gprofng collect app -O "$SCRIPT_DIR/$OUTNAME" tests/mainTest/main_test ../configs/labeled_image_config.json

