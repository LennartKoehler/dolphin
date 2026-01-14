# CMake generated Testfile for 
# Source directory: /workspace/tests/backends
# Build directory: /workspace/build/tests/backends
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(CPUBackendTest "/workspace/build/tests/backends/cpu_test")
set_tests_properties(CPUBackendTest PROPERTIES  WORKING_DIRECTORY "/workspace/build" _BACKTRACE_TRIPLES "/workspace/tests/backends/CMakeLists.txt;26;add_test;/workspace/tests/backends/CMakeLists.txt;0;")
add_test(OpenMPBackendTest "/workspace/build/tests/backends/openmp_test")
set_tests_properties(OpenMPBackendTest PROPERTIES  WORKING_DIRECTORY "/workspace/build" _BACKTRACE_TRIPLES "/workspace/tests/backends/CMakeLists.txt;48;add_test;/workspace/tests/backends/CMakeLists.txt;0;")
add_test(FFTPerformanceTest "/workspace/build/tests/backends/fft_performance_test")
set_tests_properties(FFTPerformanceTest PROPERTIES  WORKING_DIRECTORY "/workspace/build" _BACKTRACE_TRIPLES "/workspace/tests/backends/CMakeLists.txt;105;add_test;/workspace/tests/backends/CMakeLists.txt;0;")
