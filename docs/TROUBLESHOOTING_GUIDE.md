
# DOLPHIN Troubleshooting Guide

This comprehensive troubleshooting guide provides detailed solutions for common issues encountered when using DOLPHIN with the new CPU/GPU architecture. It covers installation, configuration, performance issues, runtime errors, and provides debugging strategies for users and developers alike.

## Table of Contents
- [Common Installation Issues](#common-installation-issues)
- [Configuration and Setup Problems](#configuration-and-setup-problems)
- [CPU Backend Issues](#cpu-backend-issues)
- [GPU Backend Issues](#gpu-backend-issues)
- [Configuration File Problems](#configuration-file-problems)
- [Performance Problems](#performance-problems)
- [Memory and Resource Issues](#memory-and-resource-issues)
- [Error Handling and Debugging](#error-handling-and-debugging)
- [Data and Image Issues](#data-and-image-issues)
- [GUI-Specific Problems](#gui-specific-problems)
- [Advanced Debugging Strategies](#advanced-debugging-strategies)
- [Log Analysis and Diagnosis](#log-analysis-and-diagnosis)
- [Preventive Monitoring](#preventive-monitoring)
- [Common Error Messages and Solutions](#common-error-messages-and-solutions)

## Common Installation Issues

### Issue 1: CUDA Compilation Errors

**Symptom:**
```
CMake Error at CMakeLists.txt:123 (find_package):
  Could not find CUDAToolkit!```

**Root Cause:** CUDA toolkit not properly installed or configured

**Solution:**
```bash
# Ensure CUDA is installed and detectable
nvcc --version

# If CUDA is installed but not detected:
export CUDA_PATH=/usr/local/cuda
export PATH=$CUDA_PATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH

# Verify CUDA installation
cd lib/cube
mkdir build && cd build
cmake .. -DCMAKE_CUDA_FLAGS="-arch=sm_80"  # Specify target architecture
make

# Rebuild main application
cd ../../..
mkdir build && cd build
cmake ..
make
```

**Verification Script:**
```bash
#!/bin/bash
# CUDA installation verification script

echo "=== CUDA Installation Verification ==="

# Check CUDA command tools
echo "1. CUDA Command Line Tools:"
if command -v nvcc &> /dev/null; then
    echo "✓ nvcc: $(nvcc --version | head -n1)"
else
    echo "✗ nvcc not found in PATH"
fi

# Check CUDA libraries
echo "2. CUDA Libraries:"
if [ -d "/usr/local/cuda/lib64" ]; then
    echo "✓ CUDA library directory found"
    ls /usr/local/cuda/lib64 | grep -E "libcudart|libcufft|libcublas" || echo "✗ Required CUDA libraries missing"
else
    echo "✗ CUDA library directory not found"
fi

# Check GPU availability
echo "3. GPU Hardware:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits || echo "✗ GPU query failed"
else
    echo "✗ nvidia-smi not available"
fi

# Test CUDA compilation
echo "4. CUDA Compilation Test:"
cd lib/cube
mkdir -p build_test
cd build_test
cmake .. > /dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "✓ CUDA compilation possible"
else
    echo "✗ CUDA compilation failed"
    echo "Common solutions:"
    echo "  - Install CUDA toolkit: https://developer.nvidia.com/cuda-downloads"
    echo "  - Set CUDA_PATH environment variable"
    echo "  - Check compiler architecture compatibility"
fi
```

### Issue 2: FFTW Library Not Found

**Symptom:**
```
CMake Error at CMakeLists.txt:89 (find_package):
  Could not find FFTW!```

**Root Cause:** FFTW library not installed or not detected by CMake

**Solution:**
```bash
# Install FFTW (Ubuntu/Debian)
sudo apt-get update
sudo apt-get install fftw3-dev libfftw3-bin

# Install FFTW (CentOS/RHEL)
sudo yum install fftw-devel fftw

# Install FFTW (macOS with Homebrew)
brew install fftw

# Verify FFTW installation
which fftw-wisdom
fftw-wisdom -v

# Force rebuild with FFTW detection
rm -rf build
mkdir build && cd build
cmake -DFFTW_DIR=/usr/local/lib/cmake/FFTW ..
make
```

**Alternative Manual Installation:**
```bash
# Download and install FFTW from source
cd ~
wget http://www.fftw.org/fftw-3.3.10.tar.gz
tar -xzvf fftw-3.3.10.tar.gz
cd fftw-3.3.10
./configure --enable-mpi --enable-threads --enable-shared --enable-single
make -j$(nproc)
sudo make install

# Configure environment
echo 'export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

# Test with DOLPHIN
cmake -DFFTW_DIR=/usr/local/lib/cmake/FFTW ..
```

### Issue 3: OpenCV Compilation Issues

**Symptom:**
```
CMake Error at CMakeLists.txt:145 (find_package):
  Could not find OpenCV!```

**Root Cause:** OpenCV not properly installed or version incompatible

**Solution:**
```bash
# Install OpenCV with specific version (4.6.0+ recommended)
sudo apt-get install libopencv-dev python3-opencv

# For custom OpenCV installation:
cd ~/opencv_build
wget https://github.com/opencv/opencv/archive/4.6.0.zip
unzip 4.6.0.zip
cd opencv-4.6.0
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=RELEASE \
      -DWITH_CUDA=OFF \
      -DBUILD_opencv_dnn=OFF \
      -DBUILD_opencv_videoio=OFF ..
make -j$(nproc)
sudo make install

# Configure CMake to use custom OpenCV
cmake -DOpenCV_DIR=/usr/local/lib/cmake/opencv4 ..
```

**Version Compatibility Check:**
```python
#!/usr/bin/env python3
# OpenCV version compatibility script

import cv2

print(f"OpenCV Version: {cv2.__version__}")

# Check OpenCV features
required_modules = [
    'imgcodecs', 'imgproc', 'videoio', 'highgui', 'core'
]

for module in required_modules:
    if hasattr(cv2, module):
        print(f"✓ Module '{module}' available")
    else:
        print(f"✗ Module '{module}' missing")

# Check for required functions
required_functions = [
    'imread', 'imwrite', 'cvtColor', 'resize'
]

for func_name in required_functions:
    if hasattr(cv2, func_name):
        print(f"✓ Function '{func_name}' available")
    else:
        print(f"✗ Function '{func_name}' missing")
```

### Issue 4: CMake Configuration Failures

**Symptom:**
```
CMake Error at CMakeLists.txt:XX (xxx):
Call to xxx failed.```

**Solution:**
```bash
# Clean and rebuild
rm -rf build
mkdir build && cd build

# Configure with verbose output
cmake .. -DCMAKE_VERBOSE=MAKEFILE

# Check CMake version
cmake --version

# Update CMake if needed
sudo apt-get install cmake  # Ubuntu
sudo yum install cmake     # CentOS
```

**Comprehensive Build Script:**
```bash
#!/bin/bash
# Comprehensive DOLPHIN build script set

echo "=== DOLPHIN Build Environment Check ==="

# Check system minimum requirements
echo "1. System Requirements:"
MIN_CORES=4
MIN_MEMORY=$((8 * 1024 * 1024 * 1024))  # 8GB

cores=$(nproc)
memory=$(free -k | awk '/^Mem:/{print $2}')

echo "  CPU Cores: $cores (minimum: $MIN_CORES)"
echo "  Memory: $((memory / 1024 / 1024)) MB (minimum: $((MIN_MEMORY / 1024 / 1024)) MB)"

if [ "$cores" -lt "$MIN_CORES" ]; then
    echo " ⚠  Warning: Low core count, build may be slow"
fi

if [ "$memory" -lt "$MIN_MEMORY" ]; then
    echo " ⚠  Warning: Low memory, consider increasing swap space"
fi

echo
echo "2. Dependency Status:"

check_dependency() {
    local name=$1
    local cmd=$2
    local path=$3
    local install=$4
    
    if [ -n "$cmd" ] && command -v "$cmd" &> /dev/null; then
        echo "  ✓ $name: $(eval $cmd --version 2>/dev/null | head -n1 || echo 'installed')"
        return 0
    elif [ -n "$path" ] && [ -f "$path" ]; then
        echo "  ✓ $name: installed at $path"
        return 0
    else
        echo "  ✗ $name: not found"
        [ -n "$install" ] && echo "   Install: $install"
        return 1
    fi
}

check_dependency "CMake" "cmake" "" "sudo apt-get install cmake || brew install cmake"
check_dependency "C++ Compiler" "g++" "" "sudo apt-get install build-essential"
check_dependency "CUDA Codebase" "" "/usr/local/cuda/lib64/libcudart.so" "sudo apt-get install cuda-toolkit | https://developer.nvidia.com/cuda-downloads"
check_dependency "FFTW" "" "/usr/include/fftw3.h" "sudo apt-get install fftw3-dev"
check_dependency "OpenCV" "opencv_version" "" "sudo apt-get install libopencv-dev"
check_dependency "TIFF Library" "tiffinfo" "" "sudo apt-get install libtiff-dev"

echo
echo "3. Platform Build Configuration:"

# Build sub-project first
echo "  Building lib/cube library..."
cd lib/cube
mkdir -p build && cd build

if cmake .. 2>/dev/null; then
    echo "  ✓ lib/cube configuration successful"
    if make -j$(nproc) 2>/dev/null; then
        echo "  ✓ lib/cube build successful"
    else
        echo "  ✗ lib/cube build failed"
        echo "Try: cd lib/cube/build && make VERBOSE=1"
    fi
else
    echo "  ✗ lib/cube configuration failed"
    echo "Try: cd lib/cube/build && cmake .. -DCUDA_ARCHITECTURES=\"75;80;90\""
fi

cd ../../..

# Main project configuration
echo
echo "4. Main Project Configuration:"

mkdir -p build && cd build
cmake ..

if [ $? -eq 0 ]; then
    echo "  ✓ Main project configuration successful"
    echo
    echo "5. Build Commands:"
    echo "   CPU Version:      make dolphin"
    echo "   GPU Version:      make dolphin"
    echo "   Both Versions:    make"
    echo ""
    echo "   Install (optional): sudo make install"
else
    echo " ✗ Main project configuration failed"
    echo "   Check output above for specific error details"
    echo ""
    echo "Common fixes:"
    echo "  1. Install missing dependencies listed above"
    echo "  2. Set environment variables:"
    echo "     export CUDA_PATH=/usr/local/cuda"
    echo "     export LD_LIBRARY_PATH=$CUDA_PATH/lib64:$LD_LIBRARY_PATH"
    echo "  3. Try different compiler:"
    echo "     export CC=gcc-11 && export CXX=g++-11"
fi
```

## Configuration and Setup Problems

### Issue 1: Invalid JSON Configuration

**Symptom:**
```
Error parsing configuration file: invalid JSON
Missing required field: algorithm
```

**Solution:**
```json
// Basic valid configuration template
{
  "default_algorithm": "rl",
  "default_iterations": 50,
  "gpu": "none",
  "grid": true,
  "subimageSize": 0,
  "psfSafetyBorder": 10,
  "borderType": 2,
  "time": false,
  "verbose": false,
  "algorithm": "rl"
}

// Validate configuration before use
./dolphin --validate-config config.json

# Fix common JSON issues
jq '.' config.json  # Validate and format JSON
```

**Configuration Validation Script:**
```python
#!/usr/bin/env python3
import json
import sys
import os

def validate_dolphin_config(config_path):
    """Validate DOLPHIN configuration file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Invalid JSON syntax: {e}")
        return False
    except FileNotFoundError:
        print(f"Configuration file not found: {config_path}")
        return False

    # Required fields validation
    required_fields = ["algorithm", "iterations"]
    missing_fields = [field for field in required_fields if field not in config]
    
    if missing_fields:
        print(f"Missing required fields: {missing_fields}")
        return False
    
    # Field validation
    if not isinstance(config["algorithm"], str) or not config["algorithm"]:
        print("Algorithm must be a non-empty string")
        return False
    
    if not isinstance(config["iterations"], int) or config["iterations"] <= 0:
        print("Iterations must be a positive integer")
        return False
    
    # GPU validation
    if "gpu" in config:
        valid_gpu_values = ["none", "cuda", "auto"]
        if config["gpu"] not in valid_gpu_values:
            print(f"Invalid gpu value: {config['gpu']}. Must be one of: {valid_gpu_values}")
            return False
    
    # Subimage size validation
    if "subimageSize" in config and config["subimageSize"] < 0:
        print("subimageSize cannot be negative")
        return False
    
    print("✓ Configuration file is valid")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 validate_config.py <config_path>")
        sys.exit(1)
    
    if validate_dolphin_config(sys.argv[1]):
        print("Configuration validation PASSED")
    else:
        print("Configuration validation FAILED")
        sys.exit(1)
```

### Issue 2: Image/PSF Path Resolution

**Symptom:**
```
Error: Cannot open image file - input.tif
Error: PSF file not found - psf.tif
```

**Solution:**
```bash
# Check absolute vs relative paths
pwd  # Show current working directory
ls -la input_data/  # Verify path exists

# Use absolute paths in configuration
{
  "algorithm": "rl",
  "imagePath": "/absolute/path/to/input.tif"
  "psfPath": "/absolute/path/to/psf.tif"
}

# Or use relative paths correctly
{
  "algorithm": "rl",
  "imagePath": "input_data/image.tif",
  "psfPath": "input_data/psf.tif"
}

# Test file access
./dolphin --input test_image.tif --psf test_psf.tif --verbose
```

**Path Verification Script:**
```bash
#!/bin/bash
# Path resolution script for DOLPHIN

echo "=== DOLPHIN Path Resolution Check ==="

# Check current directory
echo "Current directory:"
pwd

# Check input data
echo
echo "Input files:"
for file in "input.tif" "input/" "results/"; do
    if [ -e "$file" ]; then
        echo "✓ Found: $file"
        echo "  Size: $(du -h "$file" | cut -f1)"
        echo "  Accessible: $(if [ -r "$file" ]; then echo "Yes"; else echo "No"; fi)"
    fi
done

# Check PSF files
echo
echo "PSF files:"
for file in "psf.tif" "psf/" "psf_config.json"; do
    if [ -e "$file" ]; then
        echo "✓ Found: $file"
        echo "  Size: $(du -h "$file" | cut -f1)"
        echo "  Accessible: $(if [ -r "$file" ]; then echo "Yes"; else echo "No"; fi)"
    fi
done

# Check result directory
echo
echo "Result directory:"
if [ -d "result" ]; then
    echo "✓ Result directory exists"
    if [ -w "result" ]; then
        echo "✓ Result directory is writable"
    else
        echo "✗ Result directory is not writable"
        echo "Fix: mkdir -p result && chmod 755 result"
    fi
else
    echo "✗ Result directory not found"
    echo "Fix: mkdir -p result"
fi

# Create sample files if missing
echo
echo "Creating test files if missing:"
if [ ! -f "input.tif" ]; then
    echo "Creating sample input.tif..."
    # This would normally be your actual image file
    # dd if=/dev/urandom of=input.tif bs=1M count=10  # Create fake 10MB file
    echo "Note: Replace with actual image file"
fi

if [ ! -f "psf.tif" ]; then
    echo "Creating sample psf.tif..."
    # This would normally be your actual PSF file
    # dd if=/dev/urandom of=psf.tif bs=1M count=1   # Create fake 1MB file
    echo "Note: Replace with actual PSF file"
fi

echo
echo "Complete paths for configuration:"
echo "$(pwd)/input.tif"
echo "$(pwd)/psf.tif"
echo "$(pwd)/result/"
```

### Issue 3: Algorithm Selection Errors

**Symptom:**
```
Error: Unknown algorithm: 'invalid_algorithm'
Available algorithms: rl, rltv, rif, inverse
```

**Solution:**
```json
// Valid algorithm names in configuration
{
  "algorithm": "rl",        // Richardson-Lucy
  "algorithm": "rltv",      // Richardson-Lucy with TV
  "algorithm": "rif",       // Regularized Inverse Filter
  "algorithm": "inverse"    // Inverse Filter
}

// List available algorithms
./dolphin --list-algorithms

# Check algorithm compatibility
./dolphin --validate-config config.json --check-algorithms
```

**Algorithm Information Script:**
```python
#!/usr/bin/env python3
"""
DOLPHIN Algorithm Information and Validation Script
"""

import json
import sys

# Available algorithms and their descriptions
ALGORITHMS = {
    "rl": {
        "name": "Richardson-Lucy",
        "description": "Classic RL deconvolution algorithm",
        "gpu_supported": True,
        "recommended_iterations": 25-100,
        "use_case": "General purpose deconvolution"
    },
    "rltv": {
        "name": "Richardson-Lucy with Total Variation",
        "description": "RL algorithm with total variation regularization",
        "gpu_supported": True,
        "recommended_iterations": 50-150,
        "use_case": "Denoising and edge preservation"
    },
    "rif": {
        "name": "Regularized Inverse Filter",
        "description": "Frequency domain deconvolution with regularization",
        "gpu_supported": True,
        "recommended_iterations": 1-10,
        "use_case": "Low noise images with well-defined PSF"
    },
    "inverse": {
        "name": "Inverse Filter",
        "description": "Direct inverse filtering in frequency domain",
        "gpu_supported": False,
        "recommended_iterations": 1,
        "use_case": "Simple, unregularized inverse filtering"
    }
}

def get_algorithm_info(algorithm_name):
    """Get detailed information about an algorithm"""
    if algorithm_name not in ALGORITHMS:
        return None
    return ALGORITHMS[algorithm_name]

def validate_algorithm_config(config):
    """Validate configuration against algorithm requirements"""
    if "algorithm" not in config:
        return False, "Missing algorithm field"
    
    algorithm_name = config["algorithm"]
    algorithm_info = get_algorithm_info(algorithm_name)
    
    if algorithm_info is None:
        return False, f"Unknown algorithm: {algorithm_name}"
    
    # Check GPU compatibility
    if "gpu" not in config:
        config["gpu"] = "none"
    
    if config["gpu"] != "none" and not algorithm_info["gpu_supported"]:
        return False, f"Algorithm {algorithm_name} does not support GPU processing"
    
    # Check iteration range
    if "iterations" in config:
        iterations = config["iterations"]
        recommended = algorithm_info["recommended_iterations"]
        if isinstance(recommended, tuple):
            if iterations < recommended[0] or iterations > recommended[1]:
                print(f"Warning: {algorithm_name} typically uses {recommended[0]}-{recommended[1]} iterations")
    
    return True, f"Configuration valid for {algorithm_info['name']}"

def list_available_algorithms():
    """List all available algorithms"""
    print("Available DOLPHIN Algorithms:")
    print("-" * 50)
    
    for algo_key, algo_info in ALGORITHMS.items():
        gpu_support = "✓" if algo_info["gpu_supported"] else "✗"
        print(f"{algo_key}: {algo_info['name']} {gpu_support}")
        print(f"  Description: {algo_info['description']}")
        print(f"  Recommended iterations: {algo_info['recommended_iterations']}")
        print(f"  Use case: {algo_info['use_case']}")
        print()

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] == "--list":
        list_available_algorithms()
    else:
        config_path = sys.argv[1]
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            valid, message = validate_algorithm_config(config)
            
            if valid:
                print(f"✓ {message}")
                algorithm = get_algorithm_info(config["algorithm"])
                print(f"Algorithm: {algorithm['name']} ({config['algorithm']})")
                print(f"Description: {algorithm['description']}")
                print(f"GPU Support: {'Yes' if algorithm['gpu_supported'] else 'No'}")
                print(f"Recommended Iterations: {algorithm['recommended_iterations']}")
            else:
                print(f"✗ {message}")
                sys.exit(1)
                
        except (json.JSONDecodeError, FileNotFoundError) as e:
            print(f"Error reading configuration file: {e}")
            sys.exit(1)
```

## CPU Backend Issues

### Issue 1: FFTW Performance Degradation

**Symptom:**
```
Warning: FFTW performance degraded
Measured FFT time: Xms (expected: Yms)
```

**Solution:**
```json
// Optimized FFTW configuration
{
  "algorithm": "rltv",
  "iterations": 75,
  "gpu": "none",
  
  "cpu_optimizations": {
    "optimizePlans": true,
    "reuseExistingPlans": true,
    "planWisdomFile": "fftw_wisdom.json",
    "forceWisdomReload": false,
    "adaptiveWisdom": true
  }
}

// Generate FFTW wisdom
fftw-wisdom -v -o fftw_wisdom.json -t 32 -n

// Monitor FFTW performance
./dolphin run-config config.json --fftw-monitoring --time
```

**FFTW Performance Optimization Script:**
```bash
#!/bin/bash
# FFTW optimization script for DOLPHIN

echo "=== FFTW Performance Optimization ==="

# Generate wisdom file for common sizes
echo "1. Generating FFTW wisdom for common sizes..."
fftw-wisdom -v -o fftw_wisdom.json \
    -t 16 -n 256 -n 512 -n 1024 -n 2048 \
    -n 256,256,32 -n 512,512,64 -n 1024,1024,128

echo "✓ FFTW wisdom generated: fftw_wisdom.json"

# Check FFTW directory
echo
echo "2. FFTW configuration:"
if [ -x "$(command -v fftw-wisdom)" ]; then
    echo "FFTW wisdom tool available"
    fftw-wisdom -v
else
    echo "FFTW wisdom tool not found"
fi

# Verify FFTW libraries
echo
echo "3. FFTW libraries:"
ldconfig -p | grep fftw || echo "FFTW libraries not found in library cache"

# Performance test script
echo
echo "4. Creating FFTW performance test..."
cat > test_fftw_performance.py << 'EOF'
#!/usr/bin/env python3
import numpy as np
import time
import subprocess
import sys

def benchmark_fftw(size_x, size_y, size_z):
    """Benchmark FFTW performance for given sizes"""
    
    # Create test data
    data = np.random.rand(size_x, size_y, size_z).astype(np.float32)
    
    # Save test data
    np.savez('test_data.npz', data)
    
    cmd = [
        './dolphin',
        '--input', 'test_data.npz',
        '--fft-performance-test',
        '--size', str(size_x), str(size_y), str(size_z)
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            output = result.stdout
            
            # Extract timing information
            import re
            pattern = r'Forward FFT:(\d+)ms, Inverse FFT:(\d+)ms'
            match = re.search(pattern, output)
            
            if match:
                forward_time = int(match.group(1))
                inverse_time = int(match.group(2))
                
                fft_size = size_x * size_y * size_z / (1024.0 * 1024.0)  # MB
                forward_eff = fft_size / forward_time if forward_time > 0 else 0
                inverse_eff = fft_size / inverse_time if inverse_time > 0 else 0
                
                print(f"Size: {size_x}x{size_y}x{size_z} ({fft_size:.1f} MB)")
                print(f"  Forward FFT: {forward_time}ms ({forward_eff:.1f} MB/ms)")
                print(f"  Inverse FFT: {inverse_time}ms ({inverse_eff:.1f} MB/ms)")
                
                # Relative efficiency metric
                expected_forward = 10 + size_x * size_y * size_z / 1000000.0 * 2
                expected_inverse = 8 + size_x * size_y * size_z / 1000000.0 * 1.5
                
                forward_ratio = forward_time / expected_forward if expected_forward > 0 else 0
                inverse_ratio = inverse_time / expected_inverse if expected_inverse > 0 else 0
                
                if forward_ratio > 2.0:
                    print("  ⚠  Forward FFT significantly slower than expected")
                if inverse_ratio > 2.0:
                    print("  ⚠  Inverse FFT significantly slower than expected")
                if forward_ratio > 2.0 or inverse_ratio > 2.0:
                    print("  Try: Enable FFTW wisdom and optimizePlans")
                
                return True
            
            else:
                print("  Could not parse FFT timing from output")
                print("  Output:", output)
                return False
        else:
            print("  Command failed:", result.stderr)
            return False
            
    except subprocess.TimeoutExpired:
        print("  FFT benchmark timed out")
        return False
    except Exception as e:
        print(f"  Error running benchmark: {e}")
        return False

# Common test sizes
test_sizes = [
    (256, 256, 32),
    (512, 512, 64),
    (1024, 1024, 128)
]

print("Starting FFTW performance benchmark...")
print("Comparing with expected baseline...")
print()

for size in test_sizes:
    success = benchmark_fftw(*size)
    if not success:
        print(f"  ❌ Failed for size {size}")
    print()

# Cleanup
rm -f test_data.npz test_data.npz.*
EOF

chmod +x test_fftw_performance.py

echo
echo "5. Performance improvement suggestions:"
echo "   - Enable optimizePlans in cpu_optimizations"
echo "   - Set reuseExistingPlans to true for repetitive operations"
echo "   - UseWisdomFile and preload FFTW wisdom"
echo "   - Ensure FFTW is compiled with proper CPU optimizations"
echo "   - Compile with -march=native for best performance"
```

### Issue 2: CPU Thread Inefficiency

**Symptom:**
```
Warning: CPU thread utilization low
Thread count: 8, Effective utilization: 25%
```

**Solution:**
```json
// Optimized thread configuration
{
  "algorithm": "rltv",
  "gpu": "none",
  
  "cpu_optimizations": {
    "ompThreads": -1,              // Auto-optimal thread count
    "enableThreadPinning": true,   // CPU affinity
    "threadStrategy": "dynamic",   // Dynamic workload balancing
    "ompSchedule": "dynamic",       // Optimize for irregular workloads
    "ompChunkSize": 1024          // Balanced chunk size
  }
}

// Environment thread control
export OMP_NUM_THREADS=4         // Explicit thread count
export OMP_SCHEDULE="dynamic,64"  // Custom scheduling
export OMP_PLACES=cores         // Thread affinity

// Test different thread counts
echo "Testing different thread counts..."
for threads in 1 2 4 8; do
    export OMP_NUM_THREADS=$threads
    echo "Testing with $threads threads..."
    time ./dolphin --config test_config.json --input test.tif
done
```

**CPU Threading Optimization Script:**
```python
#!/usr/bin/env python3
"""
CPU Threading Optimization Script for DOLPHIN
"""

import os
import subprocess
import json
import multiprocessing
import time
from collections import defaultdict

def measure_cpu_performance(config, num_threads):
    """Measure execution time with thread count"""
    
    # Update configuration with thread count
    with open(config, 'r') as f:
        config_data = json.load(f)
    
    config_data['cpu_optimizations']['ompThreads'] = num_threads
    config_data['time'] = True
    
    temp_config = f'temp_config_threads_{num_threads}.json'
    with open(temp_config, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    # Set environment
    env = os.environ.copy()
    env['OMP_NUM_THREADS'] = str(num_threads)
    
    # Run benchmark
    start_time = time.time()
    result = subprocess.run(['./dolphin', '--config', temp_config, '--input', 'test_input.tif'], 
                          capture_output=True, text=True, env=env, timeout=300)
    end_time = time.time()
    
    # Cleanup
    os.remove(temp_config)
    
    if result.returncode == 0:
        duration = end_time - start_time
        return duration, result.stdout
    else:
        return None, result.stderr

def optimize_threads(config_file, test_input='test_input.tif'):
    """Find optimal thread configuration"""
    
    if not os.path.exists(config_file):
        print(f"Configuration file not found: {config_file}")
        return
    
    available_threads = multiprocessing.cpu_count()
    print(f"Available CPU cores: {available_threads}")
    print(f"Testing thread counts from 1 to min(8, {available_threads})")
    
    thread_results = []
    
    # Test different thread counts
    for threads in range(1, min(available_threads, 8) + 1):
        print(f"Testing with {threads} threads...")
        
        duration, output = measure_cpu_performance(config_file, threads)
        
        if duration is not None:
            thread_results.append((threads, duration))
            print(f"  Completed in: {duration:.2f} seconds")
            
            # Extract memory usage if available
            memory_match = None
            if 'Memory:' in output:
                for line in output.split('\n'):
                    if 'Memory:' in line:
                        memory_match = line.split('Memory:')[1].strip()
                        break
            
            if memory_match:
                print(f"  Memory: {memory_match}")
        else:
            print(f"  Failed: {output[:100]}...")
            thread_results.append((threads, float('inf')))
    
    if not thread_results:
        print("Thread testing failed for all configurations")
        return
    
    # Find optimal thread count
    thread_results.sort(key=lambda x: x[1])
    best_threads, best_time = thread_results[0]
    
    print("\n" + "="*50)
    print("THREAD PERFORMANCE RESULTS:")
    print("="*50)
    
    for threads, duration in thread_results:
        if threads == best_threads:
            marker = " (BEST)"
        else:
            marker = ""
        
        efficiency = 1.0 / duration if duration > 0 else 0
        print(f"Threads: {threads:2d} | Time: {duration:6.2f}s | Efficiency: {efficiency:8.2e}{marker}")
    
    print(f"\nOptimal thread count: {best_threads} threads")
    print(f"Best execution time: {best_time:.2f} seconds")
    
    # Calculate effective parallelism
    if len(thread_results) > 1:
        best_time = thread_results[0][1]
        single_thread_time = thread_results[-1][1]
        
        # Scaling efficiency calculated from to best time
        theoretical_best = single_thread_time / best_threads
        scaling_efficiency = theoretical_best / best_time * 100
        
        print(f"Single thread time: {single_thread_time:.2f}s")
        print(f"Theoretical best speedup: {single_thread_time/best_time:.2f}x")
        print(f"Scaling efficiency: {scaling_efficiency:.1f}%")
    
    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 20)
    
    if best_threads <= 2:
        print("✓ Single-threaded or dual-threaded processing is optimal")
        print("  This suggests memory-bound or I/O-limited workload")
    else:
        print(f"✓ {best_threads}-threaded processing is optimal")
        print("  CPU parallelization is effective")
    
    if scaling_efficiency < 80:
        print("⚠  Scaling efficiency is below 80%, consider:")
        print("  - Checking memory bandwidth limitations")
        print("  - Verifying OpenMP support is enabled")
        print("  - Testing with loading optimization")
    else:
        print("✓ Good scaling efficiency")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 optimize_threads.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    optimize_threads(config_file)
```

### Issue 3: Memory Allocation Failures

**Symptom:**
```
Error: Memory allocation failed
Failed to allocate 16GB of memory
```

**Solution:**
```json
// Memory-efficient configuration
{
  "algorithm": "rltv",
  "gpu": "none",
  "grid": true,                            // Enable grid processing
  "subimageSize": 512,                     // Optimized chunk size
  
  "cpu_optimizations": {
    "memoryPoolEnabled": true,              // Reduce allocation overhead
    "memoryPoolSize": "2GB",               // Limit memory usage
    "reuseMemoryBuffers": true,             // Buffer reuse
    "minimalValidation": true               // Reduce validation overhead
  }
}

// Memory monitoring and control
./dolphin --config config.json --memory-monitoring --memory-limit 8G

// Check current memory usage
watch -n 1 'free -h && ps aux | grep dolphin'
```

**Memory Management Diagnostic Script:**
```python
#!/usr/bin/env python3
"""
Memory Management Diagnostic for DOLPHIN
"""

import os
import subprocess
import json
import time
import psutil
import signal

def get_memory_usage():
    """Get current memory usage in human readable format"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    rss = memory_info.rss / (1024 * 1024)  # RSS in MB
    vms = memory_info.vms / (1024 * 1024)  # VMS in MB
    
    system_memory = psutil.virtual_memory()
    total_mem = system_memory.total / (1024 * 1024)
    available_mem = system_memory.available / (1024 * 1024)
    
    return {
        'rss': rss,
        'vms': vms,
        'total_system': total_mem,
        'available_system': available_mem,
        'percent_used': (1 - available_mem / total_mem) * 100
    }

def monitor_dolphin_memory(config_file, duration=60):
    """Monitor memory usage during DOLPHIN execution"""
    
    print(f"Starting DOLPHIN memory monitoring for {duration} seconds...")
    print("Configuration:", config_file)
    
    # Start DOLPHIN process
    process = subprocess.Popen(['./dolphin', '--config', config_file, '--input', 'test_input.tif'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    memory_history = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            memory_info = get_memory_usage()
            memory_history.append({
                'time': time.time() - start_time,
                **memory_info
            })
            
            print(f"Time: {memory_info['time']:.1f}s | "
                  f"RSS: {memory_info['rss']:7.1f}MB | "
                  f"System: {memory_info['percent_used']:5.1f}% | "
                  f"Available: {memory_info['available_system']:6.1f}MB")
            
            time.sleep(1)
            
            # Check if process is still alive
            if process.poll() is not None:
                break
                
    except KeyboardInterrupt:
        print("Monitoring interrupted")
    
    # Terminate process if still running
    if process.poll() is None:
        print("Terminating DOLPHIN process...")
        process.terminate()
        
        # Force kill if needed
        time.sleep(5)
        if process.poll() is None:
            process.kill()
    
    # Analyze memory usage
    analyze_memory_usage(memory_history)

def analyze_memory_usage(memory_history):
    """Analyze memory usage patterns"""
    
    if not memory_history:
        print("No memory usage data collected")
        return
    
    print("\n" + "="*60)
    print("MEMORY USAGE ANALYSIS")
    print("="*60)
    
    # Extract memory data
    rss_values = [entry['rss'] for entry in memory_history]
    system_percent = [entry['percent_used'] for entry in memory_history]
    
    max_rss = max(rss_values)
    min_rss = min(rss_values)
    avg_rss = sum(rss_values) / len(rss_values)
    
    max_system = max(system_percent)
    min_system = min(system_percent)
    avg_system = sum(system_percent) / len(system_percent)
    
    print(f"RSS Memory Usage:")
    print(f"  Maximum: {max_rss:7.1f} MB")
    print(f"  Minimum: {min_rss:7.1f} MB")
    print(f"  Average: {avg_rss:7.1f} MB")
    print(f"  Range:   {max_rss - min_rss:7.1f} MB")
    
    print(f"\nSystem Memory Usage:")
    print(f"  Maximum: {max_system:7.1f}%")
    print(f"  Minimum: {min_system:7.1f}%")
    print(f"  Average: {avg_system:7.1f}%")
    
    # Memory growth analysis
    if len(rss_values) > 10:
        first_10 = rss_values[:10]
        last_10 = rss_values[-10:]
        growth_rate = (sum(last_10) - sum(first_10)) / len(first_10)
        
        if growth_rate > 10:  # More than 10MB growth per second
            print(f"\n⚠  Memory leak detected!")
            print(f"   Growth rate: {growth_rate:.1f} MB/s")
            print(f"   Consider enabling memory pools and validation")
        else:
            print(f"\n✓ Stable memory usage")
            print(f"   Growth rate: {growth_rate:.1f} MB/s")
    
    # Memory spike detection
    memory_spikes = []
    window_size = 5
    
    for i in range(window_size, len(rss_values)):
        window = rss_values[i-window_size:i+1]
        if rss_values[i] > max(window[:-1]) * 1.2:  # 20% increase
            memory_spikes.append((i, rss_values[i]))
    
    if memory_spikes:
        print(f"\n⚠  Memory usage spikes detected:")
        for spike_time, spike_rss in memory_spikes[:3]:
            print(f"   Time {spike_time:5.1f}s: {spike_rss:7.1f} MB")
        print(f"   Total spikes: {len(memory_spikes)}")
    else:
        print(f"\n✓ No memory usage spikes detected")
    
    # Generate recommendations
    print("\nRECOMMENDATIONS:")
    print("-" * 20)
    
    if max_rss > 8096:  # More than 8GB
        print("⚠  High memory usage detected")
        print("  - Use grid processing with smaller subimages")
        print("  - Enable memory pooling")
        print("  - Consider CPU backend for memory efficiency")
    
    if len(memory_spikes) > 5:
        print("⚠  Frequent memory spikes detected")
        print("  - Check for memory leaks")
        print("  - Enable memory reinitialization")
        print("  - Consider reducing subimage size")
    
    if growth_rate > 5 and len(memory_history) > 30:  # Continuous growth detected
        print("⚠  Continuous memory growth detected")
        print("  - Memory leak likely")
        print("  - Enable verbose logging to track allocations")
        print("  - Check custom algorithm implementations")

def test_memory_efficiency_config(config_file):
    """Test different memory efficiency configurations"""
    
    print("Testing memory efficiency configurations...")
    
    # Load base configuration
    with open(config_file, 'r') as f:
        base_config = json.load(f)
    
    configs_to_test = [
        {"name": "Default", "optimization": False},
        {"name": "Memory Pool", "memoryPoolEnabled": True},
        {"name": "Grid Processing", "grid": True, "subimageSize": 512},
        {"name": "Both", "grid": True, "subimageSize": 512, "memoryPoolEnabled": True}
    ]
    
    memory_results = []
    
    for config_variant in configs_to_test:
        # Create variant configuration
        variant_config = base_config.copy()
        
        # Apply optimization settings
        if "memoryPoolEnabled" in config_variant:
            if "cpu_optimizations" not in variant_config:
                variant_config["cpu_optimizations"] = {}
            variant_config["cpu_optimizations"]["memoryPoolEnabled"] = config_variant["memoryPoolEnabled"]
        
        if "grid" in config_variant:
            variant_config["grid"] = config_variant["grid"]
        
        if "subimageSize" in config_variant:
            variant_config["subimageSize"] = config_variant["subimageSize"]
        
        # Save variant configuration
        temp_config = f"temp_config_{config_variant['name'].lower().replace(' ', '_')}.json"
        
        with open(temp_config, 'w') as f:
            json.dump(variant_config, f, indent=2)
        
        # Test configuration
        print(f"Testing {config_variant['name']} configuration...")
        
        start_memory = get_memory_usage()
        start_time = time.time()
        
        # Run DOLPHIN
        result = subprocess.run(['./dolphin', '--config', temp_config, '--input', 'test_input.tif'],
                              capture_output=True, text=True, timeout=300)
        
        end_time = time.time()
        end_memory = get_memory_usage()
        
        # Analyze result
        execution_time = end_time - start_time
        memory_peak = end_memory['rss'] - start_memory['rss']
        success = result.returncode == 0
        
        memory_results.append({
            'name': config_variant['name'],
            'execution_time': execution_time,
            'memory_peak': memory_peak,
            'success': success,
            'efficiency': 1.0 / execution_time if execution_time > 0 else 0
        })
        
        os.remove(temp_config)
    
    # Display results
    print("\n" + "="*70)
    print("MEMORY EFFICIENCY COMPARISON")
    print("="*70)
    
    print(f"{'Configuration':<20} {'Time (s)':>10} {'Peak MB':>10} {'Status':>8}")
    print("-" * 70)
    
    for result in memory_results:
        status = "✓" if result['success'] else "✗"
        print(f"{result['name']:<20} {result['execution_time']:>10.2f} "
              f"{result['memory_peak']:>10.1f} {status:>8}")
    
    # Find optimal configuration
    successful_configs = [r for r in memory_results if r['success']]
    if successful_configs:
        # Sort by combined efficiency metric
        successful_configs.sort(key=lambda x: (x['execution_time'] + x['memory_peak'] * 0.1))
        
        optimal_config = successful_configs[0]
        print(f"\n✓ Most efficient configuration: {optimal_config['name']}")
        print(f"  Execution time: {optimal_config['execution_time']:.2f}s")
        print(f"  Memory usage: {optimal_config['memory_peak']:.1f}MB")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 memory_diagnostic.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("DOLPHIN Memory Diagnostic Tool")
    print("=" * 40)
    
    choice = input("Choose mode:\n1. Monitor memory usage during execution\n2. Test efficiency configurations\n> ")
    
    if choice == "1":
        duration = input("Monitoring duration (seconds, default 60): ")
        try:
            duration = int(duration)
        except ValueError:
            duration = 60
        
        monitor_dolphin_memory(config_file, duration)
    elif choice == "2":
        test_memory_efficiency_config(config_file)
    else:
        print("Invalid choice")
        sys.exit(1)
```

## GPU Backend Issues

### Issue 1: CUDA Initialization Failure

**Symptom:**
```
Error: CUDA initialization failed
cuCtxCreate returned CUDA_ERROR_OUT_OF_MEMORY
```

**Solution:**
```json
// Reduced memory configuration for GPU
{
  "algorithm": "rltv",
  "iterations": 50,
  "gpu": "cuda",
  
  "gpu_optimizations": {
    "streamCount": 1,               // Reduce concurrent streams
    "sharedMemoryPerBlock": 8192,    // Reduce shared memory
    "blockSize": 64,                // Smaller blocks
    "memoryPoolSize": "2GB",        // Limit memory pool
    "enableTextureMemory": false    // Disable caching
  },
  
  "execution_strategy": {
    "gpu_memory_limit": "4GB",      // Explicit memory limit
    "enable_memory_monitoring": true
  }
}

// Check CUDA capabilities
nvidia-smi --query-gpu=compute_cap,memory.total,memory.used --format=csv,noheader

# Monitor GPU memory
watch -n 1 nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

**GPU Diagnostic Script:**
```python
#!/usr/bin/env python3
"""
GPU Diagnostic and Configuration Script for DOLPHIN
"""

import subprocess
import json
import os
import sys
import time

def check_cuda_availability():
    """Check CUDA/RTX availability and capabilities"""
    print("=== CUDA/GPU Availability Check ===")
    
    # Check nvidia-smi
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,memory.used', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        
        print("Available GPUs:")
        for i, line in enumerate(result.stdout.strip().split('\n')):
            name, total_mem, used_mem = line.split(',')
            total_mem = int(total_mem)
            used_mem = int(used_mem)
            free_mem = total_mem - used_mem
            available = "✓" if free_mem > 4096 else "⚠"  # At least 4GB free
            
            print(f"  GPU {i} [{available}] {name.strip()}")
            print(f"    Memory: {used_mem}MB used, {free_mem}MB free, {total_mem}MB total")
            
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ nvidia-smi not available or CUDA not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking GPUs: {e}")
        return False

def check_cuda_libraries():
    """Check CUDA libraries availability"""
    print("\n=== CUDA Libraries Check ===")
    
    cuda_libs = ['libcudart.so', 'libcublas.so', 'libcufft.so']
    lib_paths = ['/usr/local/cuda/lib64', 
                 '/usr/local/cuda/lib64/stubs', 
                 '/usr/lib/x86_64-linux-gnu', 
                 '/usr/lib64']
    
    found_libs = []
    missing_libs = []
    
    for lib in cuda_libs:
        found = False
        for path in lib_paths:
            if os.path.exists(os.path.join(path, lib)):
                found = True
                found_libs.append(f"✓ {lib} at {path}")
                break
        
        if not found:
            missing_libs.append(f"✗ {lib} not found in standard paths")
    
    if found_libs:
        print("Found CUDA libraries:")
        for lib in found_libs:
            print(f"  {lib}")
    
    if missing_libs:
        print("Missing CUDA libraries:")
        for lib in missing_libs:
            print(f"  {lib}")
        return False
    
    return len(found_libs) == len(cuda_libs)

def check_cube_dependency():
    """Check CUBE library dependency"""
    print("\n=== CUBE Library Check ===")
    
    # Try to run version check
    try:
        result = subprocess.run(['./dolphin', '--cuda-version'], 
                              capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if 'CUDA' in line or 'CUBE' in line:
                    print(f"  {line}")
            return True
        else:
            print(f"  CUBE version check failed: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print("  ✗ dolphincuda executable not found")
        return False
    except subprocess.TimeoutExpired:
        print("  ✗ CUBE version check timed out")
        return False
    except Exception as e:
        print(f"  ✗ Error checking CUBE: {e}")
        return False

def generate_gpu_config(config_file):
    """Generate GPU-optimized configuration based on available hardware"""
    
    print("\n=== Generating GPU Configuration ===")
    
    # Load base configuration
    try:
        with open(config_file, 'r') as f:
            base_config = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        print(f"Error loading configuration file: {config_file}")
        return False
    
    # Get GPU information
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        
        gpu_info = {}
        for line in result.stdout.strip().split('\n'):
            name, total_mem = line.split(',')
            gpu_info['name'] = name.strip()
            gpu_info['total_mem'] = int(total_mem)
            gpu_info['available_mem'] = int(total_mem) * 0.8  # Reserve 20%
            break  # Just use the first GPU for now
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("GPU information unavailable - using conservative defaults")
        gpu_info = {
            'name': 'Unknown',
            'total_mem': 8000,
            'available_mem': 6000
        }
    
    # Configure GPU optimizations based on hardware
    gpu_config = base_config.copy()
    
    # Enable GPU processing
    gpu_config['gpu'] = 'cuda'
    
    # Configure optimizations based on available memory
    if gpu_info['available_mem'] < 4096:  # Less than 4GB available
        print(f"Low memory GPU: {gpu_info['available_mem']}MB available")
        
        # Conservative configuration
        gpu_config['gpu_optimizations'] = {
            "usePinnedMemory": False,       # Disable to save memory
            "useAsyncTransfers": False,     # Simplify memory transfers
            "useCUBEKernels": False,       # Disable CUBE for memory savings
            "optimizePlans": True,
            "blockSize": 64,                # Smaller blocks
            "sharedMemoryPerBlock": 4096,  # Reduced shared memory
            "streamCount": 1,               # Single stream
            "enableErrorChecking": True,
            "memoryPoolSize": "1GB"        # Limited memory pool
        }
        
        gpu_config['subimageSize'] = 256  # Small subimages
        
    elif gpu_info['available_mem'] < 8192:  # 4-8GB available
        print(f"Medium memory GPU: {gpu_info['available_mem']}MB available")
        
        # Balanced configuration
        gpu_config['gpu_optimizations'] = {
            "usePinnedMemory": True,
            "useAsyncTransfers": True,
            "useCUBEKernels": True,
            "optimizePlans": True,
            "blockSize": 128,
            "sharedMemoryPerBlock": 8192,
            "streamCount": 2,
            "enableErrorChecking": True,
            "memoryPoolSize": "4GB"
        }
        
        gpu_config['subimageSize'] = 512  # Medium subimages
        
    else:  # More than 8GB available
        print(f"High memory GPU: {gpu_info['available_mem']}MB available")
        
        # Aggressive configuration
        gpu_config['gpu_optimizations'] = {
            "usePinnedMemory": True,
            "useAsyncTransfers": True,
            "useCUBEKernels": True,
            "optimizePlans": True,
            "blockSize": 256,
            "sharedMemoryPerBlock": 16384,
            "streamCount": 4,
            "enableErrorChecking": False,  # Disable for max performance
            "memoryPoolSize": "8GB"
        }
        
        gpu_config['subimageSize'] = 0  # Auto-optimal size
    
    # Save optimized configuration
    gpu_config_file = config_file.replace('.json', '_gpu_optimized.json')
    
    with open(gpu_config_file, 'w') as f:
        json.dump(gpu_config, f, indent=2)
    
    print(f"GPU-optimized configuration saved to: {gpu_config_file}")
    
    # Print summary
    print("\nGPU Configuration Summary:")
    print(f"GPU Model: {gpu_info['name']}")
    print(f"Available Memory: {gpu_info['available_mem']}MB")
    print(f"Subimage Size: {gpu_config['subimageSize']}")
    print(f"Stream Count: {gpu_config['gpu_optimizations']['streamCount']}")
    print(f"Memory Pool: {gpu_config['gpu_optimizations']['memoryPoolSize']}")
    
    return gpu_config_file

def benchmark_gpu_vs_cpu(config_file):
    """Benchmark GPU vs CPU performance"""
    
    print("\n=== GPU vs CPU Benchmark ===")
    
    if not check_cuda_availability():
        print("CUDA not available, skipping GPU benchmark")
        return
    
    # Generate CPU and GPU configurations
    gpu_config = generate_gpu_config(config_file)
    
    # Load and modify for CPU
    with open(config_file, 'r') as f:
        cpu_config = json.load(f)
    
    cpu_config['gpu'] = 'none'
    
    save_cpu_config = config_file.replace('.json', '_cpu_optimized.json')
    
    with open(save_cpu_config, 'w') as f:
        json.dump(cpu_config, f, indent=2)
    
    print(f"CPU configuration saved to: {save_cpu_config}")
    
    # Run benchmarks
    print("\nStarting performance comparison...")
    
    config_pairs = [
        ("CPU", save_cpu_config),
        ("GPU", gpu_config)
    ]
    
    results = []
    
    for name, config in config_pairs:
        print(f"\nBenchmarking {name} configuration...")
        
        start_time = time.time()
        
        try:
            result = subprocess.run(['./dolphin', '--config', config, '--input', 'test_input.tif'],
                                  capture_output=True, text=True, timeout=300)
            
            execution_time = time.time() - start_time
            
            if result.returncode == 0:
                results.append({
                    'name': name,
                    'time': execution_time,
                    'success': True,
                    'output': result.stdout
                })
                print(f"  Completed in: {execution_time:.2f} seconds")
            else:
                results.append({
                    'name': name,
                    'time': float('inf'),
                    'success': False,
                    'error': result.stderr
                })
                print(f"  Failed: {result.stderr[:100]}")
                
        except Exception as e:
            results.append({
                'name': name,
                'time': float('inf'),
                'success': False,
                'error': str(e)
            })
            print(f"  Error: {e}")
    
    # Display results
    print("\n" + "="*60)
    print("GPU vs CPU PERFORMANCE COMPARISON")
    print("="*60)
    
    successful_results = [r for r in results if r['success']]
    
    if len(successful_results) == 2:
        cpu_time = results[0]['time']
        gpu_time = results[1]['time']
        
        if cpu_time > 0 and gpu_time > 0:
            speedup = cpu_time / gpu_time
            efficiency = min(100 / speedup, 100)
            
            print(f"{'Configuration':<10} {'Time (s)':>12} {'Speedup':>12} {'GPU Efficiency':>15}")
            print("-" * 60)
            
            print(f"{'CPU':<10} {cpu_time:>12.2f} {'  1.0x':>12} {'  --':>15}")
            print(f"{'GPU':<10} {gpu_time:>12.2f} {speedup:>12.2f}x {efficiency:>13.1f}%")
            
            if speedup > 1.5:
                print("\n✓ GPU provides significant performance improvement")
                print(f"  Speedup: {speedup:.2f}x")
                print(f"  Time saved: {cpu_time - gpu_time:.2f}s ({(1-1/speedup)*100:.1f}%)")
            else:
                print("\n⚠  GPU performance improvement may not be significant")
                print("  Consider if GPU overhead justifies the speedup")
        else:
            print("Invalid timing data")
            
    elif len(successful_results) == 1:
        successful = successful_results[0]
        print(f"Only {successful['name']} completed successfully")
        print(f"The configuration failed")
        
        failed = [r for r in results if not r['success']][0]
        print(f"Failure reason: {failed['error'][:100]}")
        
    else:
        print("Both configurations failed")
        for result in results:
            print(f"  {result['name']}: {result['error'][:100]}")
    
    return results

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 gpu_diagnostic.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("DOLPHIN GPU Diagnostic Tool")
    print("=" * 40)
    
    # Run diagnostics
    cuda_ok = check_cuda_availability()
    libs_ok = check_cuda_libraries()
    cube_ok = check_cube_dependency()
    
    if cuda_ok and libs_ok and cube_ok:
        print("\n✓ GPU components appear to be properly installed")
    else:
        print("\n⚠  Some GPU components may have issues")
        if not cuda_ok:
            print("  - Check CUDA installation")
            print("  - Verify GPU drivers are up to date")
        if not libs_ok:
            print("  - Check CUDA library paths")
            print("  - Reinstall CUDA libraries")
        if not cube_ok:
            print("  - Check CUBE library compilation")
            print("  - Rebuild with CUDA support")
    
    # Ask user what to do next
    print("\nWhat would you like to do?")
    print("1. Generate GPU-optimized configuration")
    print("2. Benchmark GPU vs CPU performance")
    choice = input("> ")
    
    if choice == "1":
        generate_gpu_config(config_file)
    elif choice == "2":
        benchmark_gpu_vs_cpu(config_file)
    else:
        print("Invalid choice")
        sys.exit(1)
```

### Issue 2: GPU Memory Allocation Failures

**Symptom:**
```
Error: GPU memory allocation failed
CUDA_ERROR_OUT_OF_MEMORY: out of memory
```

**Solution:**
```json
// Memory-optimized GPU configuration
{
  "algorithm": "rl",
  "gpu": "cuda",
  "grid": true,
  "subimageSize": 256,                    // Smaller chunks reduce memory
  
  "gpu_optimizations": {
    "memoryPoolEnabled": false,           // Disable to allow system control
    "reduceMemoryOverhead": true,         // Aggressive memory optimization
    "smallMemoryStrategy": true,          // Optimized for small memory
    "enableErrorChecking": false,          // Reduce overhead
    "blockSize": 128,                     // Optimal for memory efficiency
    "sharedMemoryPerBlock": 4096,         // Reduced shared memory
    "maxMemoryUsage": "4GB"               // Explicit memory limit
  }
}

// Monitor GPU memory allocation
./dolphin --config config.json --gpu-memory-monitoring

# Check GPU memory with detailed tracking
watch -n 2 "nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits"
```

**GPU Memory Optimization Script:**
```python
#!/usr/bin/env python3
"""
GPU Memory Optimization for DOLPHIN
"""

import json
import subprocess
import time
import os
import signal

def get_gpu_memory_info():
    """Get current GPU memory information"""
    try:
        result = subprocess.run(['nvidia-smi', 
                               '--query-gpu=memory.used,memory.total,memory.free', 
                               '--format=csv,noheader,nounits'],
                              capture_output=True, text=True, check=True)
        
        used, total, free = map(int, result.stdout.strip().split('\n')[0].split(','))
        return used, total, free
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def monitor_gpu_memory_during_dolphin(config_file, duration=120):
    """Monitor GPU memory usage during DOLPHIN execution"""
    
    print(f"Starting GPU memory monitoring for {duration} seconds...")
    
    # Start DOLPHIN process
    process = subprocess.Popen(['./dolphincuda', '--config', config_file, '--input', 'test_input.tif'],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    memory_history = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            mem_info = get_gpu_memory_info()
            
            if mem_info is not None:
                used, total, free = mem_info
                memory_history.append({
                    'time': time.time() - start_time,
                    'used': used,
                    'total': total,
                    'free': free,
                    'percent_used': (used / total) * 100
                })
                
                print(f"Time: {memory_history[-1]['time']:6.1f}s | "
                      f"Used: {used:5}MB | Free: {free:5}MB | "
                      f"Total: {total:5}MB | {memory_history[-1]['percent_used']:5.1f}%")
            else:
                print(f"Time: {time.time() - start_time:6.1f}s | Unknown GPU status")
            
            time.sleep(1)
            
            # Check if process is still alive
            if process.poll() is not None:
                print("DOLPHIN process completed")
                break
                
    except KeyboardInterrupt:
        print("Monitoring interrupted by user")
    
    # Terminate process if still running
    if process.poll() is None:
        print("\nTerminating DOLPHINCUDA process...")
        process.terminate()
        
        # Force kill if needed
        time.sleep(5)
        if process.poll() is None:
            process.kill()
    
    # Analyze memory usage
    analyze_gpu_memory_usage(memory_history)

def analyze_gpu_memory_usage(memory_history):
    """Analyze GPU memory usage patterns"""
    
    if not memory_history:
        print("No GPU memory usage data collected")
        return
    
    print("\n" + "="*70)
    print("GPU MEMORY USAGE ANALYSIS")
    print("="*70)
    
    # Extract memory data
    used_memory = [entry['used'] for entry in memory_history]
    free_memory = [entry['free'] for entry in memory_history]
    percent_used = [entry['percent_used'] for entry in memory_history]
    
    max_used = max(used_memory)
    min_used = min(used_memory)
    avg_used = sum(used_memory) / len(used_memory)
    
    max_percent = max(percent_used)
    min_percent = min(percent_used)
    avg_percent = sum(percent_used) / len(percent_used)
    
    print(f"Memory Usage Statistics:")
    print(f"  Maximum: {max_used:6}MB ({max_percent:5.1f}%)")
    print(f"  Minimum: {min_used:6}MB ({min_percent:5.1f}%)")
    print(f"  Average: {avg_used:6.1f}MB ({avg_percent:5.1f}%)")
    
    # Memory analysis for various thresholds
    thresholds = [70, 80, 90]  # Percentages
    
    for threshold in thresholds:
        over_threshold = sum(1 for p in percent_used if p > threshold)
        percent_time_over = (over_threshold / len(percent_used)) * 100
        
        print(f"\nMemory usage > {threshold}% for {percent_time_over:.1f}% of time")
        print(f"  Number of samples: {over_threshold}/{len(percent_used)}")
        
        if percent_time_over > 10:
            print(f"  ⚠  High memory usage detected!")
            if percent_time_over > 30:
                print(f"     Consider: significant optimization needed")
            else:
                print(f"     Consider: moderate optimization")
    
    # Memory growth analysis
    if len(used_memory) > 20:
        first_10 = used_memory[:10]
        last_10 = used_memory[-10:]
        growth_rate = (sum(last_10) - sum(first_10)) / len(first_10)
        
        print(f"\nMemory growth rate: {growth_rate:.1f} MB/interval")
        
        if growth_rate > 20:  # Significant growth detected
            print("  ⚠  Memory leak suspected - multiple allocations without deallocation")
        elif growth_rate > 5:
            print("  ⚠  Memory gradually increasing")
        else:
            print("  ✓ Memory usage appears stable")
    
    # Peak memory analysis
    peak_threshold = 0.9  # 90% of total memory
    gpu_total = memory_history[0]['total']
    
    peak_usage_mb = max_used
    peak_threshold_mb = gpu_total * peak_threshold
    
    if peak_usage_mb > peak_threshold_mb:
        print(f"\n⚠  Memory peak ({peak_usage_mb}MB) exceeds safe threshold ({peak_threshold_mb}MB)")
        
        # Find when peak occurred
        peak_index = used_memory.index(max_used)
        peak_time = memory_history[peak_index]['time']
        
        print(f"  Peak occurred at {peak_time:.1f}s into processing")
        
        # Suggest optimizations
        print("\nOptimization suggestions:")
        print("  - Reduce subimageSize in configuration")
        print("  - Enable memoryPoolEnabled for better memory reuse")
        print("  - Disable GPU-asyncOperations to simplify memory management")
        print("  - Use fewer CUDA streams")
        print("  - Reduce shared memory usage")
        print("  - Enable aggressive memory cleanup")
    
    return {
        'max_used': max_used,
        'max_percent': max_percent,
        'growth_rate': growth_rate,
        'time_over_90p': sum(1 for p in percent_used if p > 90),
        'optimizations_needed': peak_usage_mb > peak_threshold_mb
    }

def generate_memory_optimized_config(config_file):
    """Generate memory-optimized configuration"""
    
    print("\n=== Generating Memory-Optimized GPU Configuration ===")
    
    # Load base configuration
    with open(config_file, 'r') as f:
        base_config = json.load(f)
    
    # Get current GPU memory
    mem_info = get_gpu_memory_info()
    
    if mem_info is None:
        print("Using conservative defaults - GPU information unavailable")
        available_memory = 4096  # 4GB as conservative default
    else:
        used, total, free = mem_info
        available_memory = free
    
    print(f"Available GPU Memory: {available_memory}MB")
    
    # Create memory-optimized configuration
    memory_config = base_config.copy()
    memory_config['gpu'] = 'cuda'
    
    # Adjust based on available memory
    if available_memory < 2048:
        # Very low memory system
        print("Very low memory detected - aggressive optimization needed")
        
        memory_config['gpu_optimizations'] = {
            "usePinnedMemory": False,
            "useAsyncTransfers": False,
            "useCUBEKernels": False,
            "optimizePlans": True,
            "blockSize": 64,
            "sharedMemoryPerBlock": 2048,
            "streamCount": 1,
            "enableErrorChecking": True,
            "memoryPoolEnabled": False,
            "enableTextureMemory": False,
            "enableConstantMemory": False
        }
        
        memory_config['subimageSize'] = 128  # Very small chunks
        memory_config['iterations'] = base_config.get('iterations', 50) // 2  # Reduce iterations
        
        print("Optimizations applied:")
        print("- No pinned memory")
        print("- Single stream")
        print("- Minimal shared memory")
        print("- Small subimage chunks")
        
    elif available_memory < 4096:
        # Low memory system
        print("Low memory detected - optimized for efficiency")
        
        memory_config['gpu_optimizations'] = {
            "usePinnedMemory": False,  # Disable to save memory
            "useAsyncTransfers": True,
            "useCUBEKernels": False,   # Disable for memory savings
            "optimizePlans": True,
            "blockSize": 128,
            "sharedMemoryPerBlock": 4096,
            "streamCount": 2,
            "enableErrorChecking": True,
            "memoryPoolEnabled": True,
            "memoryPoolSize": "1GB",
            "enableTextureMemory": False
        }
        
        memory_config['subimageSize'] = 256
        
        print("Optimizations applied:")
        print("- Limited pinned memory")
        print("- Enabled memory pooling")
        print("- Reduced CUBE usage")
        
    elif available_memory < 8192:
        # Medium memory system
        print("Medium memory - balanced optimization")
        
        memory_config['gpu_optimizations'] = {
            "usePinnedMemory": True,
            "useAsyncTransfers": True,
            "useCUBEKernels": True,
            "optimizePlans": True,
            "blockSize": 128,
            "sharedMemoryPerBlock": 8192,
            "streamCount": 2,
            "enableErrorChecking": False,
            "memoryPoolEnabled": True,
            "memoryPoolSize": "2GB",
            "enableTextureMemory": True
        }
        
        memory_config['subimageSize'] = 512
        
        print("Optimizations applied:")
        print("- Balanced memory pooling")
        print("- Some error checking disabled")
        print("- Medium subimage size")
        
    else:
        # High memory system
        print("High memory - aggressive optimization")
        
        memory_config['gpu_optimizations'] = {
            "usePinnedMemory": True,
            "useAsyncTransfers": True,
            "useCUBEKernels": True,
            "optimizePlans": True,
            "blockSize": 256,
            "sharedMemoryPerBlock": 16384,
            "streamCount": 4,
            "enableErrorChecking": False,
            "memoryPoolEnabled": True,
            "memoryPoolSize": "4GB",
            "enableTextureMemory": True,
            "enableConstantMemory": True,
            "enableDynamicParallelism": False
        }
        
        memory_config['subimageSize'] = 1024
        
        print("Optimizations applied:")
        print("- Aggressive streaming")
        print("- Full memory pooling")
        print("- Maximizing GPU features")
    
    # Save memory-optimized configuration
    memory_config_file = config_file.replace('.json', '_memory_optimized.json')
    
    with open(memory_config_file, 'w') as f:
        json.dump(memory_config, f, indent=2)
    
    print(f"\nMemory-optimized configuration saved to: {memory_config_file}")
    print(f"Recommended subimage size: {memory_config['subimageSize']}")
    print(f"Memory pool: {memory_config['gpu_optimizations']['memoryPoolSize']}")
    print(f"Stream count: {memory_config['gpu_optimizations']['streamCount']}")
    print(f"Block size: {memory_config['gpu_optimizations']['blockSize']}")
    
    return memory_config_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 gpu_memory_optimizer.py <config_file>")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("DOLPHIN GPU Memory Optimization Tool")
    print("=" * 50)
    
    # Check if CUBE/dolphincuda exists
    if not os.path.exists('dolphincuda'):
        print("Warning: dolphincuda not found - some features may not work")
        print("Ensure CUDA version was built with: make dolphin")
    
    choice = input("\nChoose option:\n1. Test current configuration with monitoring\n2. Generate memory-optimized configuration\n> ")
    
    if choice == "1":
        duration = input("Monitor duration (seconds, default 60): ")
        try:
            duration = int(duration)
        except ValueError:
            duration = 60
        
        monitor_gpu_memory_during_dolphin(config_file, duration)
        
    elif choice == "2":
        generate_memory_optimized_config(config_file)
        
    else:
        print("Invalid choice")
        sys.exit(1)
```

### Issue 3: GPU Usage Below 100%

**Symptom:**
```
Warning: GPU utilization below expected threshold
GPU Utilization: 45% (Expected: 80-90%)
```

**Solution:**
```json
// Enhanced GPU utilization configuration
{
  "algorithm": "rltv",
  "gpu": "cuda",
  
  "gpu_optimizations": {
    "useAsyncTransfers": true,            // Overlap transfers and computation
    "useMultipleStreams": true,           // Better workload distribution
    "optimalBlockConfig": true,           // Better block size tuning
    "enableDynamicParallelism": false,    // For some GPUs
    "prefetchData": true,                // Data prefetching
    "optimizeKernelOccupancy": true       // Better GPU utilization
  }
}

// Monitor GPU utilization
nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv -l 1

# Check kernel launch overhead
./dolphincuda --config config.json --kernel-overhead-analysis
```

**GPU Utilization Analysis Script:**
```python
#!/usr/bin/env python3
"""
GPU Utilization Analysis and Optimization for DOLPHIN
"""

import subprocess
import time
import json
import os
import matplotlib.pyplot as plt
import statistics

def get_gpu_utilization():
    """Get current GPU utilization"""
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used', '--format=csv,noheader,nounits'],
                               capture_output=True, text=True, check=True)
        
        utilization, used = map(int, result.stdout.strip().split(','))
        return utilization, used
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None

def monitor_gpu_utilization_during_dolphin(config_file, duration=60):
    """Monitor GPU utilization during DOLPHIN execution"""
    
    print(f"Starting GPU utilization monitoring for {duration} seconds...")
    
    # Start DOLPHIN process
    process = subprocess.Popen(['./dolphincuda', '--config', config_file, '--input', 'test_input.tif'],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    utilization_history = []
    memory_history = []
    start_time = time.time()
    
    try:
        while time.time() - start_time < duration:
            gpu_info = get_gpu_utilization()
            
            current_time = time.time() - start_time
            
            if gpu_info is not None:
                utilization, memory_used = gpu_info
                utilization_history.append({
                    'time': current_time,
                    'utilization': utilization,
                    'memory_used': memory_used
                })
                
                print(f"Time: {current_time:6.1f}s | GPU: {utilization:3d}% | "
                      f"Memory: {memory_used:5}MB")
                
                memory_history.append({
                    'time': current_time,
                    'memory_used': memory_used
                })
            else:
                print(f"Time: {current_time:6.1f}s | GPU Unknown Status")
            
            time.sleep(0.5)  # More frequent monitoring
            
            # Check if process is still alive
            if process.poll() is not None:
                print("DOLPHIN process completed")
                break
                
    except KeyboardInterrupt:
        print("Monitoring interrupted by user")
    
    # Clean up process
    if process.poll() is None:
        print("\nTerminating DOLPHINCUDA process...")
        process.terminate()
        time.sleep(2)
        if process.poll() is None:
            process.kill()
    
    # Analyze utilization patterns
    analyze_gpu_utilization(utilization_history, memory_history)
    
    return utilization_history, memory_history

def analyze_gpu_utilization(utilization_history, memory_history):
    """Analyze GPU utilization patterns"""
    
    if not utilization_history:
        print("No utilization data collected")
        return
    
    print("\n" + "="*70)
    print("GPU UTILIZATION ANALYSIS")
    print("="*70)
    
    # Extract data
    utilizations = [entry['utilization'] for entry in utilization_history]
    memory_values = [entry['memory_used'] for entry in memory_history]
    
    # Basic statistics
    avg_utilization = statistics.mean(utilizations)
    max_utilization = max(utilizations)
    min_utilization = min(utilizations)
    util_std = statistics.stdev(utilizations) if len(utilizations) > 1 else 0
    
    print(f"Utilization Statistics:")
    print(f"  Average: {avg_utilization:.1f}%")
    print(f"  Maximum: {max_utilization}%. Minimum: {min_utilization}%")
    print(f"  Standard deviation: {util_std:.1f}%")
    
    # Utilization efficiency assessment
    if avg_utilization < 50:
        efficiency_rating = "Poor"
        print(f"\n⚠  {efficiency_rating} GPU utilization ({avg_utilization:.1f}%)")
    elif avg_utilization < 70:
        efficiency_rating = "Fair"
        print(f"\n⚠  {efficiency_rating} GPU utilization ({avg_utilization:.1f}%)")
    elif avg_utilization < 85:
        efficiency_rating = "Good"
        print(f"\n✓ {efficiency_rating} GPU utilization ({avg_utilization:.1f}%)")
    else:
        efficiency_rating = "Excellent"
        print(f"\n✓ {efficiency_rating} GPU utilization ({avg_utilization:.1f}%)")
    
    # Utilization variance analysis
    if util_std > 20:
        print(f"\n⚠  High utilization variance detected ({util_std:.1f}%)")
        print("  Possible causes:")
        print("  - Memory transfer bottlenecks")
        print("  - Kernel launch overhead")
        print("  - Variable subimage processing times")
        print("  - Data dependency issues")
    else:
        print(f"\n✓ Stable utilization pattern")
    
    # Memory-utilization correlation
    if len(memory_values) > 10:
        memory_correlation = calculate_memory_utilization_correlation(memory_values, utilizations)
        print(f"\nMemory-Utilization Correlation: {memory_correlation:.2f}")
        
        if memory_correlation > 0.7:
            print("⚠  Strong correlation with memory usage")
            print("  Consider: Memory transfer optimization")
        else:
            print("✓ Memory usage doesn't strongly correlate with utilization")
    
    # Low utilization periods
    low_util_threshold = 30
    low_util_periods = [entry for entry in utilization_history if entry['utilization'] < low_util_threshold]
    
    if low_util_periods:
        low_util_percentage = (len(low_util_periods) / len(utilization_history)) * 100
        print(f"\nGPU utilization below {low_util_threshold}% for {low_util_percentage:.1f}% of time")
        
        if low_util_percentage > 20:
            print("⚠  Extended low utilization periods detected")
            print("  Optimization suggestions:")
            print("  - Check for memory transfer bottlenecks")
            print("  - Look for idle periods in processing")
            print("  - Consider reducing chunk size for more frequent processing")
    
    return {
        'avg_utilization': avg_utilization,
        'max_utilization': max_utilization,
        'min_utilization': min_utilization,
        'std_deviation': util_std,
        'efficiency_rating': efficiency_rating,
        'low_util_percentage': low_util_percentage if 'low_util_percentage' in locals() else 0
    }

def calculate_memory_utilization_correlation(memory_values, utilizations):
    """Calculate correlation between memory usage and GPU utilization"""
    
    if len(memory_values) != len(utilizations):
        return 0.0
    
    # Normalize values to 0-1 range
    mem_min, mem_max = min(memory_values), max(memory_values)
    util_min, util_max = min(utilizations), max(utilizations)
    
    if mem_max == mem_min or util_max == util_min:
        return 0.0
    
    normalized_memory = [(m - mem_min) / (mem_max - mem_min) for m in memory_values]
    normalized_utilization = [(u - util_min) / (util_max - util_min) for u in utilizations]
    
    # Calculate Pearson correlation coefficient
    n = len(normalized_memory)
    
    sum_products = sum(m * u for m, u in zip(normalized_memory, normalized_utilization))
    sum_memory = sum(m for m in normalized_memory)
    sum_util = sum(u for u in normalized_utilization)
    sum_memory_sq = sum(m * m for m in normalized_memory)
    sum_util_sq = sum(u * u for u in normalized_utilization)
    
    correlation_numerator = n * sum_products - sum_memory * sum_util
    correlation_denominator = ((n * sum_memory_sq - sum_memory ** 2) * 
                               (n * sum_util_sq - sum_util ** 2)) ** 0.5
    
    if correlation_denominator == 0:
        return 0.0
    
    return correlation_numerator / correlation_denominator

def generate_utility_optimized_config(config_file):
    """Generate GPU utility-optimized configuration"""
    
    print("\n=== Generating GPU Utility-Optimized Configuration ===")
    
    # Load base configuration
    with open(config_file, 'r') as f:
        base_config = json.load(f)
    
    # Utility-optimized configuration
    utility_config = base_config.copy()
    utility_config['gpu'] = 'cuda'
    
    # Enable features for better GPU utilization
    utility_config['gpu_optimizations'] = {
        "usePinnedMemory": True,
        "useAsyncTransfers": True,
        "useMultipleStreams": True,
        "useCUBEKernels": True,
        "optimizePlans": True,
        "enableDynamicParallelism": False,
        "prefetchData": True,
        "optimizeKernelOccupancy": True,
        "enableTextureMemory": True,
        "enableConstantMemory": True,
        "streamCount": 4,                     # Multiple streams
        "blockSize": 128,                     # Optimal size
        "sharedMemoryPerBlock": 8192,        # Balanced shared memory
        "enableErrorChecking": False          # Disable for better throughput
    }
    
    # Fallback strategies for better utilization
    utility_config['execution_strategy'] = {
        "enableAutoOptimization": True,
        "utilizationThreshold": 70,         # Minimum 70% target
        "adaptToHardware": True,
        "enablePrefetching": True
    }
    
    # Save utility-optimized configuration
    utility_config_file = config_file.replace('.json', '_utility_optimized.json')
    
    with open(utility_config_file, 'w') as f:
        json.dump(utility_config, f, indent=2)
    
    print(f"\nUtility-optimized configuration saved to: {utility_config_file}")
    print("\nImprovements applied:")
    print("- Enhanced async transfers and data prefetching")
    print("- Multiple CUDA streams for better overlap")
    print("- Optimized kernel occupancy configuration")
    print("- Balanced shared memory allocation")
    print("- Texture and constant memory optimizations")
    
    return utility_config_file

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python3 gpu_utilization_analyzer.py <config_file>")
        print("\nThis tool analyzes GPU utilization during DOLPHIN execution")
        sys.exit(1)
    
    config_file = sys.argv[1]
    
    print("DOLPHIN GPU Utilization Analysis Tool")
    print("=" * 60)
    
    # Check if CUBE/dolphincuda exists
    if not os.path.exists('dolphincuda'):
        print("Warning: dolphincuda not found - some features may not work")
        print("Ensure CUDA version was built with: make dolphin")
    
    choice = input("\nChoose action:\n1. Monitor utilization during execution\n2. Generate utility-optimized configuration\n> ")
    
    if choice == "1":
        duration = input("Monitor duration (seconds, default 60): ")
        try:
            duration = int(duration)
        except ValueError:
            duration = 60
        
        monitor_gpu_utilization_during_dolphin(config_file, duration)
        
    elif choice == "2":
        generate_utility_optimized_config(config_file)
        
    else:
        print("Invalid choice")
        sys.exit(1)
```

This is the first half of the comprehensive Troubleshooting Guide. I'll continue with the remaining sections to complete the full documentation.

Now let me continue with the rest of the troubleshooting guide:

## Configuration File Problems

### Issue 1: Invalid JSON Syntax

**Symptom:**
```
Error: Invalid JSON in configuration file
Expected: 'value' at line 15, column 5
```

**Solution:**
```json
// Validate JSON syntax before usage
cat config.json | python3 -m json.tool > /dev/null
if [ $? -eq 0 ]; then
    echo "✓ JSON syntax is valid"
else
    echo "✗ JSON syntax errors found"
fi

// Auto-fix common JSON issues (commas, quotes)
python3 -c "
import json
import sys

try:
    with open('config.json', 'r') as f:
        data = json.load(f)
    
    # Reformat JSON properly
    with open('config_fixed.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print('✓ Configuration fixed: config_fixed.json')
    
except json.JSONDecodeError as e:
    print(f'✗ JSON Error: {e}')
    sys.exit(1)
"
```

**JSON Validation and Repair Script:**
```python
#!/usr/bin/env python3
"""
Configuration File Validation and Repair Tool
"""

import json
import os
import sys
import re

class JSONRepairTool:
    def __init__(self):
        self.common_issues = {
            'trailing_commas': r',(?=\s*[\}\]])',
            'comments': r'//.*?$|/\*.*?\*/',
            'single_quotes': r"'([^']*)'",
            'duplicate_keys': r'"([^"]+)":\s*,\s*"([^"]+)":',
            'missing_brackets': r'\{(?!\s*})|\}(?=\s*\{)|\[(?!\s*\])|\](?=\s*\[)',
        }
    
    def validate_json_file(self, file_path):
        """Validate JSON file syntax and structure for DOLPHIN"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Basic JSON syntax check
            try:
                data = json.loads(content)
                print(f"✓ JSON syntax valid: {file_path}")
                return data, None
            except json.JSONDecodeError as e:
                line_num = e.lineno
                line_content = content.split('\n')[line_num - 1] if '\n' in content else "last line"
                
                error_info = {
                    'type': 'syntax_error',
                    'message': str(e),
                    'line': line_num,
                    'line_content': line_content,
                    'expected_start': e.msg.split(' ')[0] if ' ' in e.msg else e.msg
                }
                
                print(f"❌ JSON Error in {file_path}:")
                print(f"   Line {line_num}: {line_content}")
                print(f"   Error: {e.msg}")
                return None, error_info
                
        except FileNotFoundError:
            error_info = {
                'type': 'file_not_found',
                'message': f"File not found: {file_path}"
            }
            print(f"❌ File not found: {file_path}")
            return None, error_info
        except Exception as e:
            error_info = {
                'type': 'general_error',
                'message': str(e)
            }
            print(f"❌ Error reading {file_path}: {e}")
            return None, error_info
    
    def repair_common_json_issues(self, file_path):
        """Attempt to repair common JSON issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            original_content = content
            
            # 1. Fix trailing commas
            content = re.sub(self.common_issues['trailing_commas'], '', content, flags=re.MULTILINE)
            
            # 2. Remove comments (not valid in strict JSON)
            content = re.sub(self.common_issues['comments'], '', content, flags=re.MULTILINE | re.DOTALL)
            
            # 3. Fix single quotes to double quotes
            content = re.sub(self.common_issues['single_quotes'], r'"\1"', content)
            
            # More complex repairs require manual intervention
            repaired = False
            
            if content != original_content:
                print("✓ Applied basic JSON repairs")
                repaired = True
            
            # Try to validate the repaired JSON
            try:
                data = json.loads(content)
                print("✓ Repaired JSON is valid")
                
                # Save repaired version
                repaired_file = file_path.replace('.json', '_repaired.json')
                with open(repaired_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=2)
                
                print(f"✓ Saved repaired configuration to: {repaired_file}")
                return True, repaired_file
                
            except json.JSONDecodeError as e:
                print(f"❌ Complex repairs needed after basic fixes: {e}")
                return False, None
                
        except Exception as e:
            print(f"❌ Error during repair: {e}")
            return False, None
    
    def validate_dolphin_config_structure(self, data):
        """Validate configuration against DOLPHIN requirements"""
        errors = []
        warnings = []
        
        # Required fields
        required_fields = ["algorithm", "iterations"]
        
        for field in required_fields:
            if field not in data:
                errors.append(f"Missing required field: {field}")
        
        # Algorithm validation
        if "algorithm" in data:
            valid_algorithms = ["rl", "rltv", "rif", "inverse"]
            if data["algorithm"] not in valid_algorithms:
                errors.append(f"Invalid algorithm: {data['algorithm']}")
        
        # Iterations validation
        if "iterations" in data:
            if not isinstance(data["iterations"], int) or data["iterations"] <= 0:
                errors.append("Iterations must be a positive integer")
        
        # GPU field validation
        if "gpu" in data:
            valid_gpu_values = ["none", "cuda", "auto"]
            if data["gpu"] not in valid_gpu_values:
                errors.append(f"Invalid gpu value: {data['gpu']}")
        
        # Subimage size validation
        if "subimageSize" in data:
            subimage = data["subimageSize"]
            if not isinstance(subimage, int) or subimage < 0:
                errors.append("subimageSize must be a non-negative integer")
        
        return errors, warnings
    
    def create_template_config(self, config_type="basic"):
        """Create a template configuration file"""
        
        templates = {
            "basic": {
                "algorithm": "rltv",
                "iterations": 75,
                "lambda": 0.01,
                "gpu": "none",
                "grid": false,
                "subimageSize": 0,
                "psfSafetyBorder": 10,
                "borderType": 2,
                "time": false,
                "verbose": false
            },
            "gpu_optimized": {
                "algorithm": "rltv",
                "iterations": 100,
                "lambda": 0.015,
                "gpu": "cuda",
                "grid": true,
                "subimageSize": 0,
                "psfSafetyBorder": 10,
                "borderType": 2,
                "time": true,
                "verbose": false,
                "gpu_optimizations": {
                    "usePinnedMemory": true,
                    "useAsyncTransfers": true,
                    "useCUBEKernels": true,
                    "optimizePlans": true,
                    "enableErrorChecking": true,
                    "streamCount": 2
                }
            },
            "cpu_optimized": {
                "algorithm": "rltv",
                "iterations": 100,
                "lambda": 0.015,
                "gpu": "none",
                "grid": true,
                "subimageSize": 0,
                "psfSafetyBorder": 10,
                "borderType": 2,
                "time": true,
                "verbose": false,
                "cpu_optimizations": {
                    "ompThreads": -1,
                    "memoryPoolEnabled": true,
                    "optimizePlans": true,
                    "enableMonitoring": false
                }
            }
        }
        
        if config_type not in templates:
            print(f"❌ Unknown template type: {config_type}")
            return False
        
        template_data = templates[config_type]
        template_file = f"dolphin_template_{config_type}.json"
        
        with open(template_file, 'w') as f:
            json.dump(template_data, f, indent=2)
        
        print(f"✓ Created template configuration: {template_file}")
        print(f"  Type: {config_type}")
        print(f"  Algorithm: {template_data['algorithm']}")
        print(f"  GPU: {template_data['gpu']}")
        print(f"  File: {template_file}")
        
        return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 config_validator.py <config_file> [--fix] [--template <type>]")
        print("\nOptions:")
        print("  <config_file>       Path to configuration file")
        print("  --fix              Attempt to fix common issues")
        print("  --template <type>  Create template (basic|gpu_optimized|cpu_optimized)")
        sys.exit(1)
    
    tool = JSONRepairTool()
    action = sys.argv[1] if len(sys.argv) > 1 else None
    
    if action == "--template" and len(sys.argv) >= 3:
        # Create template configuration
        template_type = sys.argv[2]
        tool.create_template_config(template_type)
        
    elif os.path.exists(action):
        # Validate or fix configuration file
        file_path = action
        
        print(f"\n=== Validating Configuration: {file_path} ===")
        
        # Validate JSON structure
        data, error = tool.validate_json_file(file_path)
        
        if error:
            print(f"\nError type: {error['type']}")
            
            if error['type'] == 'syntax_error':
                print(f"\nRepair options:")
                print("1. Try automatic repair")
                print("2. Create valid template")
                print("3. View error details")
                
                choice = input("\nChoose option (1/2/3): ")
                
                if choice == "1":
                    success, repaired_file = tool.repair_common_json_issues(file_path)
                    if success and repaired_file:
                        print(f"\nRepaired file: {repaired_file}")
                        data, _ = tool.validate_json_file(repaired_file)
                elif choice == "2":
                    template_type = input("\nTemplate type (basic/gpu_optimized/cpu_optimized): ")
                    tool.create_template_config(template_type)
                elif choice == "3":
                    print(f"\nError details:")
                    print(f"  Line {error['line']}: {error['line_content']}")
                    print(f"  Error: {error['message']}")
                    
            elif error['type'] == 'file_not_found':
                print("Options:")
                print(" 1. Create template configuration")
                print(" 2. Check file path")
                choice = input("\nChoose option (1/2): ")
                
                if choice == "1":
                    template_type = input("\nTemplate type (basic/gpu_optimized/cpu_optimized): ")
                    tool.create_template_config(template_type)
                else:
                    print(f"Check if file exists at: {os.path.abspath(file_path)}")
            
            sys.exit(1)
        
        # Validate DOLPHIN-specific structure
        if data:
            print(f"\n=== DOLPHIN Configuration Structure ===")
            
            structure_errors, structure_warnings = tool.validate_dolphin_config_structure(data)
            
            if structure_errors:
                print("❌ Configuration structure errors:")
                for error in structure_errors:
                    print(f"   - {error}")
                return
            
            print("✓ Configuration structure is valid")
            
            if structure_warnings:
                print("⚠  Configuration warnings:")
                for warning in structure_warnings:
                    print(f"   - {warning}")
            
            # Show configuration summary
            print(f"\nConfiguration Summary:")
            print(f"  Algorithm: {data.get('algorithm', 'not specified')}")
            print(f"  GPU: {data.get('gpu', 'not specified')}")
            print(f"  Iterations: {data.get('iterations', 'not specified')}")
            print(f"  Grid processing: {data.get('grid', 'not specified')}")
            print(f"  Subimage size: {data.get('subimageSize', 'auto if 0')}")
            
            if 'gpu_optimizations' in data:
                print("  GPU optimizations: enabled")
            if 'cpu_optimizations' in data:
                print("  CPU optimizations: enabled")
        
        # Offer to generate optimized version
        if 'optimize' in sys.argv:
            print(f"\n=== Generating Optimized Configuration ===")
            print("Would you like an optimized version? (y/n): ", end="")
            response = input().lower()
            
            if response == 'y':
                tool.create_template_config("gpu_optimized")
    
    else:
        print(f"File not found: {action}")
        print("Use --template to create a new configuration")

if __name__ == "__main__":
    main()
```

### Issue 2: Parameter Type Mismatches

**Symptom:**
```
Error: Invalid parameter type: iterations must be integer, got string
Error: Invalid parameter value: subimageSize must be positive
```

**Solution:**
```json
// Correct parameter types
{
  "algorithm": "rltv",        // String
  "iterations": 100,          // Integer, not "100"
  "lambda": 0.01,            // Float, not "0.01"
  "grid": true,              // Boolean, not "true"
  "subimageSize": 512,       // Integer
  "gpu": "cuda"              // String, not cuda object
}

// Validate configuration types
python3 - "
import json

def validate_config_types(config):
    issues = []
    
    # Check known parameters with expected types
    type_checks = {
        'algorithm': str,
        'iterations': int,
        'lambda': (int, float),
        'gpu': str,
        'grid': bool,
        'subimageSize': int,
        'psfSafetyBorder': int,
        'borderType': int,
        'time': bool,
        'verbose': bool
    }
    
    for param, expected_type in type_checks.items():
        if param in config:
            actual_value = config[param]
            actual_type = type(actual_value)
            
            # Handle list/tuple types
            if isinstance(expected_type, tuple):
                if actual_type not in expected_type:
                    issues.append(f'Parameter {param}: Expected {expected_type}, got {actual_type} ({actual_value})')
            else:
                if actual_type != expected_type:
                    issues.append(f'Parameter {param}: Expected {expected_type}, got {actual_type} ({actual_value})')
    
    return issues

# Load and validate
with open('config.json', 'r') as f:
    config = json.load(f)

issues = validate_config_types(config)

if issues:
    print('Configuration type issues found:')
    for issue in issues:
        print(f'- {issue}')
else:
    print('✓ All parameter types are correct')
"

# Auto-correct simple type issues
python3 - "
import json

def auto_correct_types(config):
    corrections = []
    fixed_config = config.copy()
    
    # Common type corrections
    for param in ['iterations', 'subimageSize', 'psfSafetyBorder', 'borderType']:
        if param in fixed_config and isinstance(fixed_config[param], str):
            try:
                fixed_config[param] = int(fixed_config[param])
                corrections.append(f'Fixed {param} type: {config[param]} -> {fixed_config[param]}')
            except ValueError:
                pass
    
    # Boolean corrections
    for param in ['grid', 'time', 'verbose']:
        if param in fixed_config and isinstance(fixed_config[param], str):
            lower_val = fixed_config[param].lower()
            if lower_val in ['true', 'yes', '1']:
                fixed_config[param] = True
                corrections.append(f'Fixed {param}: {config[param]} -> True')
            elif lower_val in ['false', 'no', '0']:
                fixed_config[param] = False
                corrections.append(f'Fixed {param}: {config[param]} -> False')
    
    return corrections, fixed_config

# Load and auto-correct
with open('config.json', 'r') as f:
    config = json.load(f)

corrections, fixed_config = auto_correct_types(config)

if corrections:
    print('Auto-corrected type issues:')
    for correction in corrections:
        print(f'- {correction}')
    
    # Save corrected version
    with open('config.json', 'w') as f:
        json.dump(fixed_config, f, indent=2)
    
    print('✓ Saved corrected configuration')
else:
    print('✓ No type issues found to correct')
"
```

## Performance Problems

### Issue 1: Slow Processing Speed

**Symptom:**
```
Performance: Processing time significantly longer than expected
Warning: CPU utilization below 50% during processing
```

**Solution:**
```bash
# System-wide performance analysis
echo "=== System Performance Analysis ==="
echo "CPU Usage:"
top -l 1 | grep "CPU usage"
echo "Memory Usage:"
vm_stat | grep "Pages free"
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Status:"
    nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader,nounits
fi

# Check specific to DOLPHIN
echo "=== DOLPHIN Performance Check ==="
./dolphin --config config.json --time-performance --input test.tif

# Compare with baseline
echo "=== Performance Comparison ==="
echo "Comparing with expected baseline for image size..."
echo "Expected time range: 5-15 minutes for 1024x1024x128 image"
echo "Actual time: measure from above command"
```

**Performance Benchmark Script:**
```python
#!/usr/bin/env python3
"""
DOLPHIN Performance Benchmark and Analysis Tool
"""

import subprocess
import time
import json
import os
import sys
import statistics
import matplotlib.pyplot as plt
import numpy as np

class DOLPHINPerformanceBenchmark:
    def __init__(self):
        self.test_images = [
            ("small", 256, 256, 32),
            ("medium", 512, 512, 64),
            ("large", 1024, 1024, 128),
            ("xlarge", 2048, 2048, 256)
        ]
        
        self.baseline_times = {
            "rl": {"small": 30, "medium": 120, "large": 600, "xlarge": 2400},
            "rltv": {"small": 45, "medium": 180, "large": 900, "xlarge": 3600},
            "rif": {"small": 15, "medium": 60, "large": 300, "xlarge": 1200},
            "inverse": {"small": 10, "medium": 30, "large": 120, "xlarge": 480}
        }
    
    def create_test_image(self, width, height, depth, filename="test_image.tif"):
        """Create a test image for benchmarking"""
        
        # This would normally create a real test image
        # For now, we'll use existing files or create placeholders
        if not os.path.exists(filename):
            print(f"Creating placeholder test image: {filename}")
            # In a real implementation, this would create a synthetic image
            # with appropriate dimensions and content
        
        return filename
    
    def run_benchmark(self, config_file, algorithm, size_type, iterations=3):
        """Run a single benchmark"""
        
        width, height, depth = next(item[1:] for item in self.test_images if item[0] == size_type)
        
        # Load and modify configuration
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Set parameters for benchmark
        config['algorithm'] = algorithm
        config['time'] = True
        
        # Create temporary config file
        temp_config = f"temp_benchmark_config_{algorithm}_{size_type}.json"
        with open(temp_config, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create test image if needed
        test_image = self.create_test_image(width, height, depth)
        
        results = []
        
        for i in range(iterations):
            print(f" Run {i+1}/{iterations}...")
            
            start_time = time.time()
            
            try:
                result = subprocess.run(['./dolphin', 
                                       '--config', temp_config, 
                                       '--input', test_image],
                                      capture_output=True, text=True, timeout=600)
                
                end_time = time.time()
                duration = end_time - start_time
                
                if result.returncode == 0:
                    # Extract processing time from output
                    import re
                    time_match = re.search(r'Processing time: (\d+\.?\d*)', result.stdout)
                    processing_time = float(time_match.group(1)) if time_match else duration
                    
                    results.append({
                        'total_duration': duration,
                        'processing_time': processing_time,
                  