#!/bin/bash

# Set the directory you want to run the command from
TARGET_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTNAME="dolphin_cuda_one_thread"
# Change to that directory
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }

# sudo /usr/local/cuda-13.0/bin/nsys profile   --trace cuda,osrt,nvtx   --gpu-metrics-devices=all   --cuda-memory-usage true   --force-overwrite true   --output "$SCRIPT_DIR/$OUTNAME" ctest --verbose -R FFTPerformanceTest

sudo /usr/local/cuda-13.0/bin/nsys profile   --trace cuda,osrt,nvtx  --cpuctxsw=process-tree --gpu-metrics-devices=all   --cuda-memory-usage true   --sample=cpu --force-overwrite true   --output "$SCRIPT_DIR/$OUTNAME" ./dolphin deconvolution -c ../configs/default_config.json