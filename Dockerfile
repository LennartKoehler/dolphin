FROM dolphin-cuda-builder:latest

ARG BUILD_CUDA=ON
COPY . /workspace
WORKDIR /workspace

RUN rm -rf build && mkdir build
WORKDIR /workspace/build
RUN cmake -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_CUDA=${BUILD_CUDA} \
          -DFFTW_PATH="/usr/local/fftw3/lib" \
          -DCMAKE_GTEST_DISCOVER_TESTS_DISCOVERY_MODE=PRE_TEST \
          .. && \
    cmake --build . -- -j$(nproc) && \
    cmake --install .

WORKDIR /workspace
RUN cmake -B /workspace/cli-build -S /workspace/frontends/cli \
      -DCMAKE_BUILD_TYPE=Release && \
    cmake --build /workspace/cli-build -- -j$(nproc)
