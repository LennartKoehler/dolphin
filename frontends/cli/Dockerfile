# Base image
FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

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
# Layer 7: Download FFTW3
# -----------------------
WORKDIR /build
RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz

# -----------------------
# Layer 8: Extract FFTW3
# -----------------------
RUN tar -xzf fftw-3.3.10.tar.gz

# -----------------------
# Layer 9: Build and install FFTW3 (single precision)
# -----------------------
WORKDIR /build/fftw-3.3.10
RUN ./configure \
        --enable-float \
        --enable-threads \
        --enable-static \
        --disable-shared \
        --prefix=/usr/local/fftw3 && \
    make -j$(nproc) && \
    make install

# -----------------------
# Upgrade GCC to 13 (Ubuntu 22.04 ships GCC 11 with incomplete C++20 library support)
# -----------------------
RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y gcc-13 g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 && \
    rm -rf /var/lib/apt/lists/*

# -----------------------
# Copy project
# -----------------------
WORKDIR /workspace
COPY . /workspace

# -----------------------
# Build and install the dolphin library
# -----------------------
RUN rm -rf /workspace/dolphin_build && mkdir /workspace/dolphin_build
WORKDIR /workspace/dolphin_build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_CUDA=ON \
          -DFFTW_PATH="/usr/local/fftw3/lib" \
          .. && \
    cmake --build . -- -j$(nproc) && \
    cmake --install .

# -----------------------
# Build the CLI frontend
# -----------------------
RUN rm -rf /workspace/cli_build && mkdir /workspace/cli_build
WORKDIR /workspace/cli_build
# Configure the CLI as a standalone consumer of the *installed* dolphin
# package (built in the previous stage). The source dir must point at
# frontends/cli, NOT at the project root — otherwise CMake re-runs the
# whole library build from source, ignores find_package(dolphin), and
# falls back to the hardcoded FFTW_PATH default (/usr/local/fftw-release/lib).
# The installed dolphinTargets.cmake already carries the absolute FFTW
# library paths, so no FFTW_PATH is needed here.
RUN cmake -DCMAKE_PREFIX_PATH="/usr/local" \
          ../frontends/cli && \
    cmake --build . -- -j$(nproc)

# -----------------------
# Set entrypoint
# -----------------------
WORKDIR /workspace
ENTRYPOINT ["./cli_build/dolphin_cli"]
