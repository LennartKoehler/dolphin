# Building Dolphin Frontends

This directory contains the frontend executables for Dolphin. Each frontend is a standalone CMake project that links against the Dolphin shared library.

## Prerequisites

1. **Build the Dolphin library first**:
   ```bash
   cd ../../
   mkdir build && cd build
   cmake ..
   make dolphin  # This builds the shared library
   ```

## Building the CLI Frontend

```bash
cd cli/
mkdir build && cd build
cmake ..
make
```

This will create the `dolphin_cli` executable.

## Building the GUI Frontend

```bash
cd gui/
mkdir build && cd build
cmake ..
make
```

This will create the `dolphin_gui` executable.

## Running

- **CLI**: `./dolphin_cli --help`
- **GUI**: `./dolphin_gui`

## Architecture

Each frontend is a complete standalone application that:
- Links against the `libdolphin.so` shared library
- Has its own main() function
- Manages its own dependencies
- Can be built and distributed independently

The dolphin library provides the core deconvolution functionality, while the frontends provide different user interfaces (command-line vs. graphical).
