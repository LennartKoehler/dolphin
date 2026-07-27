#!/usr/bin/env bash
set -e

IMAGE=dolphin-cuda-builder:latest

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building builder image..."
    docker build -f Dockerfile.builder -t "$IMAGE" .
fi

echo "Building project + CLI in container..."
docker run --rm \
    -v "$PWD:/workspace" \
    "$IMAGE" \
    bash -c '
        cmake \
            -S /workspace \
            -B /workspace/dockerbuild/dolphin_build \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/usr/local \
            -DBUILD_CUDA=ON \
            -DFFTW_PATH=/usr/local/fftw3/lib

        cmake --build /workspace/dockerbuild/dolphin_build --parallel
        cmake --install /workspace/dockerbuild/dolphin_build

        cmake \
            -S /workspace/frontends/cli \
            -B /workspace/dockerbuild/cli_build \
            -DCMAKE_PREFIX_PATH=/usr/local

        cmake --build /workspace/dockerbuild/cli_build --parallel
    '
