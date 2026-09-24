#!/bin/bash
set -e

echo "Building project + CLI in container..."
docker build -f Dockerfile -t dolphin_build .

echo "Extracting build artifacts..."
docker create --name tempcontainer dolphin_build
docker cp tempcontainer:/workspace/build ./dockerbuild
docker cp tempcontainer:/workspace/cli-build ./dockerbuild/cli-build
docker rm tempcontainer

echo "Build artifacts in ./dockerbuild/"
