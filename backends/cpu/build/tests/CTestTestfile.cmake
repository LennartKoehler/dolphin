# CMake generated Testfile for 
# Source directory: /home/lennart-k-hler/projects/dolphin/backends/cpu/tests
# Build directory: /home/lennart-k-hler/projects/dolphin/backends/cpu/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(cpuBackendTest "cuda_test")
set_tests_properties(cpuBackendTest PROPERTIES  _BACKTRACE_TRIPLES "/home/lennart-k-hler/projects/dolphin/backends/cpu/tests/CMakeLists.txt;36;add_test;/home/lennart-k-hler/projects/dolphin/backends/cpu/tests/CMakeLists.txt;0;")
add_test(MultipleDevices "/home/lennart-k-hler/projects/dolphin/backends/cpu/build/tests/multiple_devices_test")
set_tests_properties(MultipleDevices PROPERTIES  _BACKTRACE_TRIPLES "/home/lennart-k-hler/projects/dolphin/backends/cpu/tests/CMakeLists.txt;38;add_test;/home/lennart-k-hler/projects/dolphin/backends/cpu/tests/CMakeLists.txt;0;")
