#!/bin/bash

# Enhanced CUDA debugging script for DOLPHIN
# This script demonstrates how to get detailed CUDA error information

echo "=== Enhanced CUDA Debugging Script ==="
echo

# Check if CUDA is properly installed
echo "1. Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc found: $(which nvcc)"
    nvcc --version | head -3
else
    echo "✗ nvcc not found"
fi

if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found: $(which nvidia-smi)"
    echo "GPU Information:"
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "✗ nvidia-smi not found"
fi

echo
echo "2. Testing CUDA runtime..."
# Create a simple test program
cat > test_cuda.cu << 'EOF'
#include <cuda_runtime.h>
#include <iostream>
#include <string>

void checkCUDAError(cudaError_t err, const std::string& operation, const std::string& context = "") {
    if (err != cudaSuccess) {
        std::cerr << "[CUDA ERROR] " << cudaGetErrorString(err) 
                  << " | Operation: " << operation 
                  << " | Context: " << context << std::endl;
        
        // Get additional info
        int device;
        cudaGetDevice(&device);
        std::cerr << "[CUDA INFO] Device: " << device << std::endl;
        
        cudaDeviceProp props;
        cudaGetDeviceProperties(&props, device);
        std::cerr << "[CUDA INFO] Device name: " << props.name << std::endl;
        
        throw std::runtime_error("CUDA error");
    }
}

int main() {
    try {
        std::cout << "Testing CUDA runtime..." << std::endl;
        
        // Test device selection
        int device_count;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        checkCUDAError(err, "cudaGetDeviceCount", "Getting device count");
        
        std::cout << "Found " << device_count << " CUDA devices" << std::endl;
        
        if (device_count == 0) {
            std::cerr << "No CUDA devices found!" << std::endl;
            return 1;
        }
        
        // Set device 0
        err = cudaSetDevice(0);
        checkCUDAError(err, "cudaSetDevice", "Setting device 0");
        
        // Test memory allocation
        size_t size = 256 * 256 * 64 * sizeof(double) * 2; // 256x256x64 complex numbers
        void* ptr;
        err = cudaMalloc(&ptr, size);
        checkCUDAError(err, "cudaMalloc", "Allocating " + std::to_string(size) + " bytes");
        
        std::cout << "Successfully allocated " << (size / 1024 / 1024) << " MB of GPU memory" << std::endl;
        
        // Test cuFFT
        #include <cufft.h>
        cufftHandle plan;
        err = cufftCreate(&plan);
        if (err != CUFFT_SUCCESS) {
            std::cerr << "[CUFFT ERROR] Failed to create plan: " << err << std::endl;
        } else {
            std::cout << "Successfully created cuFFT plan" << std::endl;
            cufftDestroy(plan);
        }
        
        // Free memory
        err = cudaFree(ptr);
        checkCUDAError(err, "cudaFree", "Freeing GPU memory");
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
EOF

# Compile and run the test
if command -v nvcc &> /dev/null; then
    echo "3. Compiling and running CUDA test..."
    nvcc -o test_cuda test_cuda.cu -lcuda -lcufft
    
    if [ $? -eq 0 ]; then
        echo "✓ Compilation successful"
        echo "4. Running CUDA test..."
        ./test_cuda
        rm -f test_cuda test_cuda.cu
    else
        echo "✗ Compilation failed"
        rm -f test_cuda test_cuda.cu
    fi
else
    echo "✗ Cannot compile test without nvcc"
    rm -f test_cuda test_cuda.cu
fi

echo
echo "=== Manual Debugging Steps ==="
echo "To get detailed error information when running your DOLPHIN application:"
echo
echo "1. Set environment variables for better error reporting:"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   export CUDA_ERROR_CHECKING=1"
echo "   export CUDA_DEVICE_DEBUG=1"
echo
echo "2. Run your application with these commands:"
echo "   cd /path/to/dolphin/build"
echo "   export CUDA_LAUNCH_BLOCKING=1"
echo "   ./tests/mainTest/main_test ../../assets/default_config.json"
echo
echo "3. For additional debugging, use:"
echo "   cuda-gdb ./tests/mainTest/main_test"
echo "   (gdb) run ../../assets/default_config.json"
echo
echo "4. Check GPU memory usage:"
echo "   nvidia-smi -l 1"
echo
echo "5. Monitor for memory leaks:"
echo "   watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv"