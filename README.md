<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>DOLPHIN</h1>
</div>


---

Deconvolution with Optimized Local PSFs for High-speed Image recoNstruction (DOLPHIN) is a C++ command-line tool designed for deconvolution of microscopy images. It supports multiple deconvolution algorithms and allows the generation and use of synthetic Point Spread Functions (PSF). The tool is intended for users familiar with image processing and deconvolution techniques.

## Features

- **Input Image Formats**: Supports both single image files and directories of image slices in TIF format.
- **Point Spread Function (PSF) Input**: Allows users to provide a PSF as a file, a directory of slices, or generate a synthetic PSF.
- **Multiple Deconvolution Algorithms**:
    - Richardson-Lucy (RL)
    - Richardson-Lucy with Total Variation (RLTV)
    - Richardson-Lucy with Adaptive Damping (RLAD)
    - Regularized Inverse Filter (RIF)
    - Inverse Filter
    - Convolution
- **Support for multiple PSFs**: Users can provide or generate multiple PSFs for specific layers or subimages.
- **Flexible Parameters**: Adjustable parameters such as sigma values for synthetic PSF generation, iteration counts, lambda for regularization, and more.
- **Image Subdivision**: Processes images as grids of smaller subimages for memory efficiency and better performance.
- **Time Measurement**: Option to display the duration of deconvolution processes.
- **Configuration via CLI or JSON**: Users can specify parameters through command-line arguments or by providing a JSON configuration file.

## Requirements

Standard usage
- C++20 or later
- [FFTW](http://www.fftw.org/) 3.3.10 (for fast Fourier transforms) 
- [LibTIFF](https://libtiff.gitlab.io/libtiff/) 4.7.0 (for Tag Image File Format)
- [ITK](https://itk.org/) (for image processing)

GPU acceleration
- [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) 12.8+ (NVIDIA CUDA Compiler Driver)
- [cuFFT](https://docs.nvidia.com/cuda/cufft/) (for fast Fourier transforms on GPU)
  
included Header files
- [CLI11](https://github.com/CLIUtils/CLI11) (for command-line parsing)
- [nlohmann/json](https://github.com/nlohmann/json) (for JSON handling)
- [spdlog](https://github.com/gabime/spdlog) (for logging)
- [CUBE](https://git.uni-jena.de/qi36soq/cube) (for operations on GPU)

## Build

```bash
mkdir ./build
cd ./build

cmake ..
make
```

The CUBE library (for GPU support) is built automatically via `add_subdirectory`. Use `-DBUILD_CUDA=ON` (default) or `-DBUILD_CUDA=OFF` to control GPU support.

## Usage

### Command-Line Interface

DOLPHIN uses a subcommand-based CLI with two modes:

```
dolphin psfgenerator    Generate PSF file
dolphin deconvolution   Run deconvolution
```

Both subcommands accept configuration via JSON files or individual CLI flags:

```
-c, --config <path>          Path to combined configuration file
-s, --setup_config <path>    Path to setup config JSON file
-d, --deconv_config <path>   Path to deconvolution config JSON file (deconvolution only)
-p, --psf_configs <paths>    Path(s) to PSF config JSON file(s)
```

Algorithm names must match exactly: `RichardsonLucy`, `RichardsonLucyTotalVariation`,
`RichardsonLucywithAdaptiveDamping`, `RegularizedInverseFilter`, `InverseFilter`, `Convolution`.

### Example

```bash
./dolphin deconvolution -c config.json
```

This command will run deconvolution using a combined JSON configuration file.

```bash
./dolphin psfgenerator -s setup_config.json -p psf_config.json
```

This command will generate a PSF using the specified setup and PSF configuration files.

### Using a Configuration File

You can specify your input, PSF, and other parameters using JSON files. See the
example configurations in `configs_checkpoint/` for reference.

```bash
./dolphin deconvolution -c config.json
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses the CLI11 library for command-line argument parsing.
- The `nlohmann/json` library is used for reading and handling JSON files.
- `FFTW` is used for fast Fourier transformations during the deconvolution process.
- `ITK` is used for image processing.
- `spdlog` is used for logging.
- Icon attribution <a href="https://www.flaticon.com/free-icons/whale" title="whale icons">Whale icons created by Freepik - Flaticon</a>

---

## Contact

For questions or feedback, please contact [christoph.manitz@uni-jena.de].

---

