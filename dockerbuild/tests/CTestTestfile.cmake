# CMake generated Testfile for 
# Source directory: /workspace/tests
# Build directory: /workspace/build/tests
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(DolphinTests "/workspace/build/tests/dolphin_tests")
set_tests_properties(DolphinTests PROPERTIES  WORKING_DIRECTORY "/workspace/build" _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;30;add_test;/workspace/tests/CMakeLists.txt;0;")
add_test(FeatheringTest "/workspace/build/tests/feathering_test")
set_tests_properties(FeatheringTest PROPERTIES  _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;48;add_test;/workspace/tests/CMakeLists.txt;0;")
add_test(TiffGenerationTest "/workspace/build/tests/tiff_generation_test")
set_tests_properties(TiffGenerationTest PROPERTIES  WORKING_DIRECTORY "/workspace/build" _BACKTRACE_TRIPLES "/workspace/tests/CMakeLists.txt;70;add_test;/workspace/tests/CMakeLists.txt;0;")
subdirs("backends")
subdirs("mainTest")
