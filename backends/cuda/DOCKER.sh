#!/bin/bash
# Build the image
docker build -t cuda .

# Create a container (without running it)
docker create --name tempcontainer cuda

# Copy the binary from container to host
docker cp tempcontainer:/workspace/build ./dockerbuild

# Remove the temporary container
docker rm tempcontainer
