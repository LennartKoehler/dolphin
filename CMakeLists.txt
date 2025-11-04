cmake_minimum_required(VERSION 3.10)
project(Deconvtool)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -ffast-math -fopenmp -march=native")
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
# Include directories and libraries for FFTW, OpenCV, OpenMP, and TIFF
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
include_directories(/usr/local/include)
# link_directories(/usr/local/lib)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(TIFF REQUIRED)


# # Compiler flags for Valgrind-safe build
# if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
#     add_compile_options(
#         -O0             # Disable optimizations
#         -g              # Debug symbols
#         -march=x86-64   # Only baseline instructions
#         -mno-avx        # Disable AVX
#         -mno-avx2       # Disable AVX2
#         -mno-fma        # Disable FMA
#         -Wall           # Enable warnings
#         -Wextra
#     )
# endif()

# Find CUDA toolkit if available (optional)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
    message(STATUS "CUDA toolkit found: ${CUDAToolkit_VERSION}")
    set(CUDA_AVAILABLE TRUE)
else()
    message(STATUS "CUDA toolkit not found. Building CPU-only version.")
    set(CUDA_AVAILABLE FALSE)
endif()



# IMGUI-------------------------------------------
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)   # Needs GLFW installed system-wide
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLEW REQUIRED glew)


set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/lib/imgui)


add_library(imgui_lib
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_opengl3.cpp
)
target_link_libraries(imgui_lib PUBLIC
    OpenGL::GL
    glfw
    ${GLEW_LIBRARIES}
)
target_include_directories(imgui_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/imgui
    ${CMAKE_SOURCE_DIR}/lib/imgui/backends
    ${GLEW_INCLUDE_DIRS}
)

# IMPLOT3 and imguiFileExplorer------------------------------
set(IMPLOT3D_DIR ${CMAKE_SOURCE_DIR}/lib/implot3d)

add_library(implot3d_lib
    ${IMPLOT3D_DIR}/implot3d.cpp
    ${IMPLOT3D_DIR}/implot3d_items.cpp

)

target_include_directories(implot3d_lib PUBLIC
    ${CMAKE_SOURCE_DIR}/lib/implot3d
    ${CMAKE_SOURCE_DIR}/lib/imgui-filebrowser)
    
target_link_libraries(implot3d_lib PUBLIC
    imgui_lib)



# Common include dirs
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TIFF_INCLUDE_DIR}
    ${CMAKE_SOURCE_DIR}/include
)
# ------------------------------------------------------------
# Collect sources
# ------------------------------------------------------------

set(BACKEND_SOURCES
    src/backend/ComplexData.cpp)
add_library(backendlib ${BACKEND_SOURCES})


set(COMMON_SOURCES
    src/Dolphin.cpp
    src/PSFCreator.cpp
    src/ServiceFactory.cpp
    src/PSFGenerationService.cpp
    src/DeconvolutionService.cpp
    src/ThreadPool.cpp
    # src/backend/ComplexData.cpp
    src/backend/BackendFactory.cpp
    
    src/hyperstack/HyperstackImage.cpp
    src/hyperstack/Image3D.cpp
    src/hyperstack/ImageMetaData.cpp
    src/hyperstack/HyperstackIO.cpp
    src/hyperstack/HyperstackMetaData.cpp

    src/psf/PSF.cpp
    src/psf/generators/GaussianPSFGenerator.cpp
    src/psf/generators/BornWolfModel.cpp
    src/psf/generators/GibsonLanniPSFGenerator.cpp
    src/psf/generators/SimpsonIntegrator.cpp
    src/psf/configs/PSFConfig.cpp
    src/psf/configs/GibsonLanniPSFConfig.cpp
    src/psf/configs/GaussianPSFConfig.cpp

    src/deconvolution/DeconvolutionConfig.cpp
    src/deconvolution/DeconvolutionAlgorithmFactory.cpp
    src/deconvolution/Preprocessor.cpp
    src/deconvolution/Postprocessor.cpp
    src/deconvolution/DeconvolutionProcessor.cpp

    src/deconvolution/algorithms/RLDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLTVDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/RLADDeconvolutionAlgorithm.cpp
    src/deconvolution/algorithms/TestAlgorithm.cpp
    
    # src/utilities/UtlFFT.cpp
    src/utilities/UtlGrid.cpp
    src/utilities/UtlImage.cpp
    src/utilities/UtlIO.cpp

    src/frontend/CLIFrontend.cpp
    src/frontend/gui/imguiWidget.cpp
    src/frontend/gui/DefaultGUIStyleConfig.cpp
    src/frontend/gui/GUIFrontend.cpp
    src/frontend/SetupConfig.cpp



    src/frontend/gui/windows/SpecificWindows.cpp
    src/frontend/gui/windows/MainWindow.cpp
    src/frontend/gui/windows/FunctionContent.cpp
    src/frontend/gui/windows/Window.cpp
    src/frontend/gui/windows/ConfigWindow.cpp
    src/frontend/gui/windows/BackendConfigWindow.cpp

    src/frontend/gui/SpinningDonut.cpp
)
add_library(dolphinlib ${COMMON_SOURCES})
# ------------------------------------------------------------
# Build shared library containing all algorithms
# ------------------------------------------------------------

target_link_libraries(dolphinlib
    PUBLIC
        imgui_lib
        implot3d_lib
        ${OpenCV_LIBS}
        ${TIFF_LIBRARIES}
        OpenMP::OpenMP_CXX

)

target_include_directories(dolphinlib
    PUBLIC
        ${CMAKE_SOURCE_DIR}/include

)

# ------------------------------------------------------------
# CPU executable
# ------------------------------------------------------------
add_executable(dolphin src/main.cpp)
target_link_libraries(dolphin
    PRIVATE
        dolphinlib
        backendlib
        pthread
        gomp
        stdc++fs)

add_subdirectory(backends/cpu)

# ------------------------------------------------------------
# GPU executable (only if CUDA found)
# ------------------------------------------------------------
if(CUDA_AVAILABLE)
    enable_language(CUDA)
    message(STATUS "CUDA found, building dolphincuda with CUBE.")
    add_compile_definitions(CUDA_AVAILABLE)
    add_subdirectory(backends/cuda)
    # Add CUBE subproject
    # target_link_libraries(cuda_backend
    #     PUBLIC
    #         dolphinlib  # cudalib depends on dolphinlib headers
    # )

else()
    message(STATUS "CUDA not found. Building only CPU version.")
endif()
