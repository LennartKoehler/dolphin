#include <iostream>
#include <memory>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include <thread>
#include "CUDABackend.h"

void testCUDABackendInitialization() {
}

int main() {
    std::cout << "Starting CUDA Backend Test" << std::endl;
    testCUDABackendInitialization();
    return 0;
}