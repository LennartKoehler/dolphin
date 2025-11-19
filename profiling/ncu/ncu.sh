#!/bin/bash

# Set the directory you want to run the command from
TARGET_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTNAME="matmul"
# Change to that directory
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }

sudo /usr/local/cuda-13.0/bin/ncu -f -o  "$SCRIPT_DIR/$OUTNAME" --target-processes all ctest --verbose -R FFTPerformanceTest 
