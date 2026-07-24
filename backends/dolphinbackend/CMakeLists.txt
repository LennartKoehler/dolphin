set(SOURCES
    src/ComplexData.cpp
    src/IBackendMemoryManager.cpp
)

add_library(dolphinbackend STATIC ${SOURCES})

# On macOS, the Apple linker does a single pass over static archives.
# Tell CMake to repeat this library in the link line so symbols are
# resolved regardless of archive ordering.
set_target_properties(dolphinbackend PROPERTIES LINK_INTERFACE_MULTIPLICITY 3)

target_link_libraries(dolphinbackend PRIVATE project_options)
# Statically link libstdc++ and libgcc so binaries run on older systems (GCC/Linux only)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND NOT APPLE)
    target_link_options(dolphinbackend PUBLIC -static-libstdc++ -static-libgcc)
endif()
# target_compile_options(dolphinbackend PRIVATE
#     -g
#     -O0
#     -fno-omit-frame-pointer
# )

target_include_directories(dolphinbackend
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include/dolphinbackend
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
