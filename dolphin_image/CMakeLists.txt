cmake_minimum_required(VERSION 3.10)
project(dolphin_image VERSION 1.0.0)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ------------------------------------------------------------
# dolphin_image: Image handling library (Image3D, TIFF IO, padding, operations)
# ------------------------------------------------------------

set(DOLPHIN_IMAGE_SOURCES
    src/Image3D.cpp
    src/ImageMetaData.cpp
    src/ImagePadding.cpp
    src/ImageOperations.cpp
    src/TiffReader.cpp
    src/TiffWriter.cpp
    src/TiffReaderWriterPair.cpp
)

add_library(dolphin_image STATIC ${DOLPHIN_IMAGE_SOURCES})

target_link_libraries(dolphin_image PUBLIC
    dolphinbackend
    ${TIFF_LIBRARIES}
    ${ITK_LIBRARIES}
)
target_link_libraries(dolphin_image PRIVATE project_options)

target_include_directories(dolphin_image
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../lib/nlohmann/json/single_include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../lib/spdlog/include/>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/../backends/dolphinbackend/include/>
        $<INSTALL_INTERFACE:include>
        ${TIFF_INCLUDE_DIR}
        ${ITK_INCLUDE_DIRS}
)
