# GUI Frontend CMakeLists.txt
cmake_minimum_required(VERSION 3.10)
project(DolphinGUI)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)




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

# Find the dolphin package
find_package(dolphin REQUIRED)

# Add OpenGL and GLFW dependencies
find_package(OpenGL REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(GLEW REQUIRED glew)

# Create GUI executable
add_executable(dolphin_gui
    src/main.cpp
    src/GUIFrontend.cpp
    src/DefaultGUIStyleConfig.cpp
    src/windows/Window.cpp
    src/windows/ConfigWindow.cpp
    src/windows/MainWindow.cpp
    src/windows/SpecificWindows.cpp
    src/windows/FunctionContent.cpp
    src/windows/BackendConfigWindow.cpp
    src/imguiWidget.cpp
    src/SpinningDonut.cpp
)

# Include directories
target_include_directories(dolphin_gui
    PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib/imgui
        ${CMAKE_SOURCE_DIR}/lib/imgui/backends
        ${CMAKE_SOURCE_DIR}/lib/implot3d
        ${CMAKE_SOURCE_DIR}/lib/imgui-filebrowser
        ${GLEW_INCLUDE_DIRS}
)
# Link against the Dolphin library and required dependencies
target_link_libraries(dolphin_gui
    PRIVATE
        dolphin::dolphin
        imgui_lib
        implot3d_lib
        OpenGL::GL
        glfw
        ${GLEW_LIBRARIES}
)

