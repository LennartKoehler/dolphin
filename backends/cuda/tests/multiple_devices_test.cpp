#include <iostream>
#include <memory>
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/CuboidShape.h"
#include <thread>
#include "dolphinbackend/IBackend.h"
#include "CUDABackend.h"
void testMultipleDevices() {
}

int main() {
    std::cout << "Starting CUDA Backend Test" << std::endl;
    testMultipleDevices();
    return 0;
}