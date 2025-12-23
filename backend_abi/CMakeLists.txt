# Create backend library
set(BACKEND_SOURCES
    src/backend/ComplexData.cpp)
add_library(backendlib ${BACKEND_SOURCES})

target_include_directories(backendlib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include)

