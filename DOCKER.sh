#!/bin/bash
# Build the image



docker build -t dolphin_build .

# Create a container (without running it)
docker create --name tempcontainer dolphin_build

# Copy the binary from container to host
docker cp tempcontainer:/workspace/build ./dockerbuild


# Remove the temporary container
docker rm tempcontainer
