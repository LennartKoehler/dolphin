# CUDA Error Debugging Guide for DOLPHIN

## Problem Analysis

Based on your error output:
```
[DEBUG] Successfully created cuFFT plans for shape: 256x256x64
CUDA error: invalid argument
CUDA error: an illegal memory access was encountered
run.sh: line 5: 3229860 Segmentation fault      (core dumped) ./tests/mainTest/main_test ../../assets/default_config.json
```

You're experiencing two main issues:
1. **CUDA invalid argument error** - This typically occurs when passing invalid parameters to CUDA functions
2. **Illegal memory access** - This indicates accessing memory outside allocated bounds or using invalid pointers
3. **Segmentation fault** - The application crashes due to memory corruption

## Why You're Not Getting Detailed Error Information

The current error handling in `CUDABackend.h` uses basic macros that only show:
- Basic error string (e.g., "invalid argument")
- Generic operation name
- No context about parameters, memory state, or call stack

## Solutions

### 1. Immediate Fix: Environment Variables

Set these environment variables before running your application to get better error reporting:

```bash
export CUDA_LAUNCH_BLOCKING=1          # Makes CUDA synchronous for better error detection
export CUDA_ERROR_CHECKING=1           # Enables additional error checking
export CUDA_DEVICE_DEBUG=1              # Enables device-level debugging
export CUDA_VISIBLE_DEVICES=0          # Force using GPU 0
```

### 2. Enhanced Error Handling

I've created `CUDAErrorUtils.h` with enhanced error checking. Here's how to use it:

```cpp
// Instead of:
CUDA_CHECK(err, "operation");

// Use:
CUDA_CHECK_EX(err, "operation", "additional context like array size, dimensions, etc.");

// Example:
CUDA_CHECK_EX(cudaMalloc(&ptr, size), "cudaMalloc", 
              "Allocating " + std::to_string(size) + " bytes for 256x256x64 array");
```

### 3. Memory Debugging

Add memory checks before critical operations:

```cpp
// Check memory state before allocation
dolphin::backend::cuda_utils::printMemoryInfo("Before allocation");

// Check for errors after operations
if (!dolphin::backend::cuda_utils::checkCUDAErrorSilent()) {
    std::cerr << "CUDA error detected!" << std::endl;
}
```

### 4. Specific Debugging Steps for Your Error

The "invalid argument" error with cuFFT plans suggests:

1. **Plan size mismatch**: The FFT plan was created for 256x256x64, but you're trying to use it with different dimensions
2. **Invalid data pointers**: The input/output pointers might be invalid or misaligned
3. **Memory corruption**: Previous operations might have corrupted memory

### 5. Debugging Commands

Run these commands to diagnose the issue:

```bash
# Check GPU status
nvidia-smi

# Monitor memory usage during execution
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv

# Run with enhanced error reporting
export CUDA_LAUNCH_BLOCKING=1
export CUDA_ERROR_CHECKING=1
./tests/mainTest/main_test ../../assets/default_config.json

# For memory debugging
export CUDA_MEMCHECK=1
./tests/mainTest/main_test ../../assets/default_config.json
```

### 6. Code Changes Needed

Update your `CUDABackend.cpp` to use enhanced error checking:

```cpp
// In forwardFFT function, replace:
CUFFT_CHECK(cufftExecZ2Z(forward, reinterpret_cast<cufftDoubleComplex*>(in.data), 
                        reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD), "forwardFFT");

// With:
CUFFT_CHECK_EX(cufftExecZ2Z(forward, reinterpret_cast<cufftDoubleComplex*>(in.data), 
                           reinterpret_cast<cufftDoubleComplex*>(out.data), FFTW_FORWARD), 
               "forwardFFT", 
               "Input size: " + std::to_string(in.size.width) + "x" + 
               std::to_string(in.size.height) + "x" + std::to_string(in.size.depth));
```

### 7. Common Causes for Your Specific Error

1. **Plan size mismatch**: The FFT plan was created for specific dimensions, but data with different dimensions is being processed
2. **Memory allocation failure**: Previous allocations might have failed silently
3. **Stream synchronization issues**: Asynchronous operations might not be properly synchronized
4. **Invalid data pointers**: Pointers might be pointing to freed memory or invalid locations

### 8. Prevention Strategies

1. **Always validate data sizes** before FFT operations
2. **Check memory allocation success** immediately after allocation
3. **Use proper stream synchronization** when mixing synchronous and asynchronous operations
4. **Add bounds checking** for array accesses
5. **Use memory debugging tools** like cuda-memcheck or valgrind

### 9. Testing Your Fix

After implementing the enhanced error handling, run:

```bash
# Run the debug script
./enhanced_cuda_debug.sh

# Test with your application
export CUDA_LAUNCH_BLOCKING=1
export CUDA_ERROR_CHECKING=1
./tests/mainTest/main_test ../../assets/default_config.json
```

### 10. Additional Tools

For more advanced debugging:

```bash
# Use cuda-gdb for debugging
cuda-gdb ./tests/mainTest/main_test
(gdb) run ../../assets/default_config.json

# Use Nsight Systems for profiling
nsys profile -o profile ./tests/mainTest/main_test ../../assets/default_config.json

# Use cuda-memcheck for memory errors
cuda-memcheck ./tests/mainTest/main_test ../../assets/default_config.json
```

## Summary

The key to getting better error information is:
1. **Use environment variables** for immediate error detection
2. **Implement enhanced error checking** with context information
3. **Add memory monitoring** throughout the application
4. **Validate all parameters** before CUDA calls
5. **Use proper debugging tools** for deeper analysis

This approach will help you identify exactly where and why the CUDA errors are occurring, rather than just seeing generic error messages.