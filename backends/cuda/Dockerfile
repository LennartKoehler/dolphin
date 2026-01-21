# Use NVIDIA CUDA + Ubuntu 22.04 (glibc 2.35)
FROM nvidia/cuda:12.8.0-devel-ubuntu22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace
COPY . /workspace
RUN rm -rf build && mkdir -p build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Debug ..
# Copy your project
COPY src/ src/
COPY include/ include/


# Build your project

RUN cmake --build build -- -j$(nproc)
