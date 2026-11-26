#!/usr/bin/env bash
# Build Dolphin inside the CUDA builder container (.github/docker/Dockerfile).
# Source is bind-mounted; artifacts land in ./dockerbuild/dolphin_build on the host.
set -euo pipefail

IMAGE=dolphin-cuda-builder:latest
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building builder image..."
    docker build -f "$ROOT/.github/docker/Dockerfile" --target builder -t "$IMAGE" "$ROOT"
fi

echo "Building project + CLI in container..."
docker run --rm -v "$ROOT:/workspace" "$IMAGE" bash -c '
    cmake -S /workspace -B /workspace/dockerbuild/dolphin_build \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA=ON \
        -DFFTW_PATH=/usr/local/fftw3/lib \
        -DCMAKE_INSTALL_PREFIX=/usr/local
    cmake --build /workspace/dockerbuild/dolphin_build --parallel
'

echo "Build artifacts in ./dockerbuild/"
