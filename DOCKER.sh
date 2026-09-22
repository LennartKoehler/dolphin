#!/bin/bash
set -e

IMAGE=dolphin-cuda-builder:latest

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    echo "Building builder image..."
    docker build -f Dockerfile.builder -t "$IMAGE" .
fi

echo "Building project + CLI in container..."
docker build -f Dockerfile -t dolphin_build .

echo "Extracting build artifacts..."
docker create --name tempcontainer dolphin_build
docker cp tempcontainer:/workspace/build ./dockerbuild
docker cp tempcontainer:/workspace/cli-build ./dockerbuild/cli-build
docker rm tempcontainer

echo "Build artifacts in ./dockerbuild/"
