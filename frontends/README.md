# Building Dolphin Frontends

This directory contains the frontend executables for Dolphin. Both frontends are built from the root CMakeLists.txt (CLI) or separately (GUI).

## Prerequisites

Build the Dolphin library first from the project root:
```bash
cd ../../
mkdir build && cd build
cmake ..
make
```

## Building the CLI Frontend

The CLI frontend is built automatically as part of the main build:
```bash
mkdir build && cd build
cmake ..
make
```

This will create the `dolphin` executable.

## Building the GUI Frontend

The GUI frontend is built separately:
```bash
cd gui/
mkdir build && cd build
cmake ..
make
```

This will create the `dolphin_gui` executable.

## Running

- **CLI**: `./dolphin --help`
- **GUI**: `./dolphin_gui`

## Architecture

Each frontend is a standalone application that:
- Links against the dolphin static library
- Has its own main() function
- The CLI is integrated into the root CMake build
- The GUI is built separately and links against ImGui, ImPlot3D
