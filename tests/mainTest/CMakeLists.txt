# Test CMakeLists.txt
cmake_minimum_required(VERSION 3.10)


set(CMAKE_BUILD_TYPE Debug)

enable_testing()

add_executable(main_test
    main_test.cpp)

target_link_options(main_test PRIVATE
    -static-libstdc++
    -static-libgcc
)
target_link_libraries(main_test PRIVATE
    -Wl,-Bstatic m mvec -Wl,-Bdynamic
)
target_link_libraries(main_test
    PRIVATE
        dolphin)


add_test(NAME MainTest COMMAND main_test)
set_tests_properties(MainTest PROPERTIES WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})

