# CMake generated Testfile for 
# Source directory: /workspace/tests
# Build directory: /workspace/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CUDABackendTest "/workspace/build/tests/cuda_test")
set_tests_properties(CUDABackendTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;36;add_test;/workspace/tests/CMakeLists.txt;0;")
add_test(MultipleDevices "/workspace/build/tests/multiple_devices_test")
set_tests_properties(MultipleDevices PROPERTIES  _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;38;add_test;/workspace/tests/CMakeLists.txt;0;")
