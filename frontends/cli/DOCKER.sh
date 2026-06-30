#!/bin/bash
# Build the image
docker build -t dolphin_cli -f Dockerfile ../..

# Remove stale dockerbuild directory
rm -rf ./dockerbuild

# Create a container (without running it)
docker create --name tempcontainer dolphin_cli

# Copy the binary from container to host
docker cp tempcontainer:/workspace/dolphin_build ./dockerbuild
docker cp tempcontainer:/workspace/cli_build ./dockerbuild

# Remove the temporary container
docker rm tempcontainer
