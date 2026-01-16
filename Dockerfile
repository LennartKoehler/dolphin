# Base image
FROM ubuntu:22.04

# Install build tools
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    && rm -rf /var/lib/apt/lists/*

# -----------------------
# Directories
# -----------------------
WORKDIR /build

# -----------------------
# Layer 1: Download ITK
# -----------------------
RUN wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.4.5/InsightToolkit-5.4.5.tar.gz

# -----------------------
# Layer 2: Extract ITK
# -----------------------
RUN tar -xzf InsightToolkit-5.4.5.tar.gz

# -----------------------
# Layer 3: Build and install ITK
# -----------------------
WORKDIR /build/InsightToolkit-5.4.5
RUN mkdir ITK-build
WORKDIR /build/InsightToolkit-5.4.5/ITK-build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_TESTING=OFF \
          -DBUILD_EXAMPLES=OFF \
          ../ && \
    make -j$(nproc) && \
    make install

# -----------------------
# Layer 4: Download libtiff
# -----------------------
WORKDIR /build
RUN wget http://download.osgeo.org/libtiff/tiff-4.5.1.tar.gz

# -----------------------
# Layer 5: Extract libtiff
# -----------------------
RUN tar -xzf tiff-4.5.1.tar.gz

# -----------------------
# Layer 6: Build and install libtiff
# -----------------------
WORKDIR /build/tiff-4.5.1
RUN ./configure \
        --disable-jbig \
        --disable-webp \
        --disable-lzma \
        --disable-libdeflate \
        --disable-zlib \
        --disable-zstd \
        --disable-lz4 \
        --disable-docs \
        --disable-tools \
        --disable-tests \
        --enable-static \
        --disable-shared \
        --prefix=/usr/local && \
    make -j$(nproc) && \
    make install

# -----------------------
# Copy project
# -----------------------
WORKDIR /workspace
COPY . /workspace

# -----------------------
# Build your project
# -----------------------
WORKDIR /workspace
RUN rm -rf build && mkdir build
WORKDIR /workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Debug .. && \
    cmake --build . -- -j$(nproc)
