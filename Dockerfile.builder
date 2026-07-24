FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN wget http://download.osgeo.org/libtiff/tiff-4.5.1.tar.gz && \
    tar -xzf tiff-4.5.1.tar.gz
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

WORKDIR /build
RUN wget https://github.com/InsightSoftwareConsortium/ITK/releases/download/v5.4.5/InsightToolkit-5.4.5.tar.gz && \
    tar -xzf InsightToolkit-5.4.5.tar.gz
WORKDIR /build/InsightToolkit-5.4.5
RUN mkdir ITK-build
WORKDIR /build/InsightToolkit-5.4.5/ITK-build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DBUILD_TESTING=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DModule_ITKIOTIFF=OFF \
          ../ && \
    make -j$(nproc) && \
    make install

WORKDIR /build
RUN wget https://www.fftw.org/fftw-3.3.10.tar.gz && \
    tar -xzf fftw-3.3.10.tar.gz
WORKDIR /build/fftw-3.3.10
RUN ./configure \
        --enable-float \
        --enable-threads \
        --enable-static \
        --disable-shared \
        --prefix=/usr/local/fftw3 && \
    make -j$(nproc) && \
    make install

RUN apt-get update && apt-get install -y software-properties-common && \
    add-apt-repository ppa:ubuntu-toolchain-r/test && \
    apt-get update && apt-get install -y gcc-13 g++-13 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-13 130 && \
    update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-13 130 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]
