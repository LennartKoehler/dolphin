cmake_minimum_required(VERSION 3.10)
project(dolphinbackend VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(SOURCES
    src/ComplexData.cpp)

add_library(dolphinbackend STATIC ${SOURCES})


target_include_directories(dolphinbackend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/dolphinbackend>
        )



# Install library and headers
install(TARGETS dolphinbackend
    EXPORT dolphinbackendTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

install(DIRECTORY include/dolphinbackend/
    DESTINATION include/dolphinbackend
)

# Export CMake targets for find_package
install(EXPORT dolphinbackendTargets
    FILE dolphinbackendTargets.cmake
    NAMESPACE dolphinbackend::
    DESTINATION lib/cmake/dolphinbackend
)

include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinbackendConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/dolphinbackendConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinbackendConfig.cmake"
    @ONLY
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinbackendConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/dolphinbackendConfigVersion.cmake"
    DESTINATION lib/cmake/dolphinbackend
)