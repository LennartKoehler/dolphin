# CMake generated Testfile for 
# Source directory: /home/lennart-k-hler/projects/dolphin/backends/cuda/tests
# Build directory: /home/lennart-k-hler/projects/dolphin/backends/cuda/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CUDABackendTest "/home/lennart-k-hler/projects/dolphin/backends/cuda/build/tests/cuda_test")
set_tests_properties(CUDABackendTest PROPERTIES  _BACKTRACE_TRIPLES "/home/lennart-k-hler/projects/dolphin/backends/cuda/tests/CMakeLists.txt;36;add_test;/home/lennart-k-hler/projects/dolphin/backends/cuda/tests/CMakeLists.txt;0;")
add_test(MultipleDevices "/home/lennart-k-hler/projects/dolphin/backends/cuda/build/tests/multiple_devices_test")
set_tests_properties(MultipleDevices PROPERTIES  _BACKTRACE_TRIPLES "/home/lennart-k-hler/projects/dolphin/backends/cuda/tests/CMakeLists.txt;38;add_test;/home/lennart-k-hler/projects/dolphin/backends/cuda/tests/CMakeLists.txt;0;")
