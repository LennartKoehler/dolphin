#!/bin/bash

# Set the directory you want to run the command from
TARGET_DIR="build"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTNAME="test"
cd "$TARGET_DIR" || { echo "Failed to change directory to $TARGET_DIR"; exit 1; }
valgrind --tool=callgrind --dump-instr=yes --collect-jumps=yes --log-file=../profiling/valgrind/"$OUTNAME" ./dolphin deconvolution -c ../configs/default_config.json

