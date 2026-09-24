<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>DOLPHIN</h1>
</div>


---

Deconvolution with Optimized Local PSFs for High-speed Image recoNstruction (DOLPHIN) is a C++ command-line tool designed for deconvolution of microscopy images. It supports multiple deconvolution algorithms and allows the generation and use of synthetic Point Spread Functions (PSF). The tool is intended for users familiar with image processing and deconvolution techniques.

## Features

- **Input Image Formats**: Supports both single image files and directories of image slices in TIF format.
- **Point Spread Function (PSF) Input**: Allows users to provide a PSF as a file, a directory of slices, or generate a synthetic PSF.
- **PSF Models**:
    - Gaussian — tunable sigma values and quality factor
    - Gibson-Lanni — physically-based model with optical design/experimental parameters
- **Multiple Deconvolution Algorithms**:
    - Richardson-Lucy (RL)
    - Richardson-Lucy with Total Variation (RLTV)
    - Richardson-Lucy with Adaptive Damping (RLAD)
    - Regularized Inverse Filter (RIF)
    - Inverse Filter
    - Convolution
- **Labeled Image Deconvolution**: Assign different PSFs to specific regions of the image via a labeled image and label-to-PSF mapping.
- **Support for multiple PSFs**: Users can provide or generate multiple PSFs for specific layers or subimages.
- **Flexible Padding**: Configurable padding fill types (zero, mirror, linear, quadratic, sinusoid, gaussian) and padding strategies (none, parent, full_psf, manual).
- **Output Compression**: TIFF output supports none, LZW, and Deflate compression with adjustable compression level.
- **GPU Acceleration**: Optional CUDA backend for GPU-accelerated FFT operations.
- **Image Subdivision**: Processes images as grids of smaller subimages for memory efficiency and better performance.
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

### Library (CPU + GPU backends)

```bash
mkdir ./build
cd ./build

cmake ..
make
```

### CLI Executable

The CLI frontend is **off by default**. Enable it explicitly:

```bash
mkdir ./build
cd ./build

cmake .. -DBUILD_CLI=ON
make
```

This produces the `./dolphin` executable.

### Tests

```bash
cd ./build
cmake .. -DENABLE_TESTS=ON -DBUILD_CLI=ON
make
ctest --output-on-failure
```

### Build Options

| Option | Default | Description |
|---|---|---|
| `BUILD_CUDA` | `ON` | Build CUDA GPU backend (requires CUDA Toolkit 12.8+) |
| `BUILD_CLI` | `OFF` | Build the CLI frontend executable (`dolphin`) |
| `ENABLE_TESTS` | `ON` | Build GoogleTest suite and register with ctest |
| `ENABLE_BENCHMARKS` | `ON` | Build benchmark executables |
| `BUILD_DOLPHIN_LIBRARY` | `ON` | Build the main dolphin static library |

The CUBE library (for GPU support) is built automatically via `add_subdirectory` when `BUILD_CUDA=ON`.

## Usage

### Command-Line Interface

DOLPHIN uses a subcommand-based CLI with two modes:

```
dolphin psfgenerator    Generate PSF file
dolphin deconvolution   Run deconvolution
```

Both subcommands accept configuration via JSON files or individual CLI flags. When a JSON config and CLI flags overlap, the JSON config takes precedence and the CLI flags are ignored.

#### Config File Flags
See configs_checkpoints directory for examples of configuration files.

```
-c, --config <path>          Path to combined configuration file (setup + deconvolution + PSF)
-s, --setup_config <path>    Path to setup config JSON file
-d, --deconv_config <path>   Path to deconvolution config JSON file (deconvolution only)
-p, --psf_configs <paths>    Path(s) to PSF config JSON file(s)
```

#### Key CLI Flags

All configuration parameters are also exposed as CLI flags. The most commonly used ones:

**Setup:**
```
-i, --image_path <path>       Input image path (TIF file or directory)
-o, --output <path>           Output path
    --backend <cpu|gpu>       Compute backend
    --n_worker_threads <n>    Number of worker threads
    --n_io_threads <n>        Number of I/O threads
    --n_devices <n>           Number of GPU devices
    --save_psf                Save the used PSF to disk
    --psf_file_paths <paths>  Path(s) to pre-existing PSF file(s)
    --labeled_image <path>    Labeled image for region-based PSF assignment
    --label_psf_map <map>     Label-to-PSF mapping (e.g. "0[ID1234], 1[constant_hyperstack_1]")
    --output_compression <type>      TIFF compression: none, lzw, deflate
    --output_compression_level <n>   Compression level (-1 to 9)
```

**Deconvolution:**
```
    --algorithm_name <name>    Algorithm (see list below)
    --iterations <n>           Number of iterations
    --epsilon <val>            Convergence threshold
    --lambda <val>             Regularization parameter
    --padding_fill <type>      Padding fill: zero, mirror, linear, quadratic, sinusoid, gaussian
    --padding_strategy <type>  Padding strategy: none, parent, full_psf, manual
    --feathering_radius <n>   Feathering radius for subimage boundaries
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

### JSON Configuration

A combined JSON config file contains three top-level sections: `setup_config`, `deconvolution_config`, and `psf_configs`. Example configurations are in `configs_checkpoint/`.

#### Annotated Example

```json
{
  "deconvolution_config": {
    "algorithm_name": "RichardsonLucy",
    "iterations": 30,
    "epsilon": 1e-6,
    "lambda": 0.015,
    "padding_fill": "mirror",
    "padding_strategy": "parent"
  },

  "psf_configs": [
    {
      "model_name": "Gaussian",
      "id": "ID1234",
      "res_lateral_nm": 5000,
      "res_axial_nm": 5000,
      "size_x": 64,
      "size_y": 64,
      "size_z": 64,
      "sigma_x": 5,
      "sigma_y": 5,
      "sigma_z": 5
    }
  ],

  "setup_config": {
    "image_path": "/path/to/input.tif",
    "output": "/path/to/output.tif",
    "backend": "cpu",
    "n_io_threads": 2,
    "n_worker_threads": 11,
    "n_devices": 1,
    "max_mem_host_gb": 64,
    "max_mem_device_gb": 8,
    "psf_file_paths": [],
    "save_psf": true,
    "output_compression": "none",
    "output_compression_level": -1
  }
}
```

#### `setup_config` Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `image_path` | string | yes | — | Input image path (TIF file or directory of slices) |
| `output` | string | yes | — | Output image path |
| `backend` | string | no | `cpu` | Compute backend (`cpu` or `gpu`) |
| `n_threads` | int | no | `1` | Number of threads |
| `n_worker_threads` | int | no | `1` | Number of worker threads |
| `n_io_threads` | int | no | `1` | Number of I/O threads |
| `n_devices` | int | no | `1` | Number of GPU devices |
| `max_mem_host_gb` | float | no | `0` | Maximum host memory in GB (0 = unlimited) |
| `max_mem_device_gb` | float | no | `0` | Maximum device memory in GB (0 = unlimited) |
| `psf_file_paths` | string[] | no | `[]` | Path(s) to pre-existing PSF file(s) |
| `save_psf` | bool | no | `false` | Save the used PSF to disk |
| `labeled_image` | string | no | — | Labeled image path for region-based PSF assignment |
| `label_psf_map` | string | no | — | Label-to-PSF mapping (e.g. `"0[ID1234], 1[constant_hyperstack_1]"`) |
| `output_compression` | string | no | `none` | TIFF compression: `none`, `lzw`, `deflate` |
| `output_compression_level` | int | no | `-1` | Compression level (-1 = default, 0–9 for deflate) |
| `tile_width` | int | no | `0` | TIFF tile width (0 = strips) |
| `tile_length` | int | no | `0` | TIFF tile length (0 = strips) |

#### `deconvolution_config` Parameters

| Parameter | Type | Required | Default | Description |
|---|---|---|---|---|
| `algorithm_name` | string | yes | `RichardsonLucy` | Algorithm selection (see names above) |
| `iterations` | int | no | `10` | Number of iterations (1–10000) |
| `epsilon` | float | no | `1e-6` | Convergence threshold (1e-12 to 1e-3) |
| `lambda` | float | no | `0.001` | Regularization parameter (0–1) |
| `padding_fill` | string | no | `zero` | Fill method: `zero`, `mirror`, `linear`, `quadratic`, `sinusoid`, `gaussian` |
| `padding_strategy` | string | no | `parent` | Strategy: `none`, `parent`, `full_psf`, `manual` |
| `padding_relative_max` | float | no | `0` | Pad until PSF is below this fraction of max PSF value |
| `feathering_radius` | int | no | `0` | Feathering radius for subimage boundaries |
| `cube_padding` | int[3] | no | `[-1,-1,-1]` | Manual padding per axis (x, y, z); doubled internally |

#### `psf_configs`

An array of PSF config objects. Each must contain a `model_name` field. See [PSF Models](#psf-models) below for model-specific parameters.

#### Labeled Image Deconvolution

When `labeled_image` is set in `setup_config`, DOLPHIN switches to labeled deconvolution mode. Each unique label value in the labeled image is mapped to a PSF via `label_psf_map`. Only the labeled region's outline is deconvolved with its assigned PSF. Set `labeled_image` and `label_psf_map` in `setup_config` to enable this mode (see `configs_checkpoint/default_config.json`).

### PSF Models

#### Gaussian

A simple Gaussian PSF defined by sigma values along each axis.

```json
{
  "model_name": "Gaussian",
  "id": "ID1234",
  "size_x": 64,
  "size_y": 64,
  "size_z": 64,
  "sigma_x": 5,
  "sigma_y": 5,
  "sigma_z": 5,
  "quality_factor": 1.0,
  "res_lateral_nm": 5000,
  "res_axial_nm": 5000,
  "NA": 1.0
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | string | — | Identifier used in `label_psf_map` |
| `size_x`, `size_y`, `size_z` | int | `20`, `20`, `10` | PSF dimensions in voxels |
| `sigma_x`, `sigma_y`, `sigma_z` | float | `10.0` | Gaussian sigma per axis |
| `quality_factor` | float | `1.0` | Blur factor (1.0 = ideal, >1 = blurrier) |
| `res_lateral_nm` | float | `200.0` | Lateral resolution in nm |
| `res_axial_nm` | float | `200.0` | Axial resolution in nm |
| `NA` | float | `1.0` | Numerical aperture |
| `nanometer_scale` | float | `1e-9` | Nanometer scale factor |
| `pixel_scaling` | float | `1e-6` | Pixel scaling factor |

#### Gibson-Lanni

A physically-based PSF model accounting for microscope optical design and experimental deviations (cover slip thickness, immersion medium refractive index mismatches, etc.).

```json
{
  "model_name": "GibsonLanni",
  "id": "GL_PSF",
  "size_x": 64,
  "size_y": 64,
  "size_z": 64,
  "NA": 1.4,
  "res_lateral_nm": 100,
  "res_axial_nm": 100,
  "lambda_nm": 520.0,
  "working_distance_design_nm": 150000.0,
  "working_distance_experimental_nm": 150000.0,
  "immersion_ri_design": 1.515,
  "immersion_ri_experimental": 1.515,
  "coverslip_thickness_design_nm": 170.0,
  "coverslip_thickness_experimental_nm": 170.0,
  "coverslip_ri_design": 1.5,
  "coverslip_ri_experimental": 1.5,
  "sample_ri": 1.33,
  "particle_axial_position_nm": 1000.0,
  "pixel_size_axial_nm": 100.0,
  "pixel_size_lateral_nm": 100.0,
  "OVER_SAMPLING": 4.0,
  "accuracy": 32
}
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `id` | string | — | Identifier used in `label_psf_map` |
| `size_x`, `size_y`, `size_z` | int | `20`, `20`, `10` | PSF dimensions in voxels |
| `NA` | float | `1.0` | Numerical aperture |
| `res_lateral_nm` | float | `200.0` | Lateral resolution in nm |
| `res_axial_nm` | float | `200.0` | Axial resolution in nm |
| `lambda_nm` | float | `520.0` | Emission wavelength in nm |
| `working_distance_design_nm` | float | `150000.0` | Design working distance (objective) in nm |
| `working_distance_experimental_nm` | float | `150000.0` | Experimental working distance in nm |
| `immersion_ri_design` | float | `1.515` | Design immersion medium refractive index |
| `immersion_ri_experimental` | float | `1.515` | Experimental immersion medium refractive index |
| `coverslip_thickness_design_nm` | float | `170.0` | Design coverslip thickness in nm |
| `coverslip_thickness_experimental_nm` | float | `170.0` | Experimental coverslip thickness in nm |
| `coverslip_ri_design` | float | `1.5` | Design coverslip refractive index |
| `coverslip_ri_experimental` | float | `1.5` | Experimental coverslip refractive index |
| `sample_ri` | float | `1.33` | Sample refractive index |
| `particle_axial_position_nm` | float | `1000.0` | Axial position of the particle in nm |
| `pixel_size_axial_nm` | float | `100.0` | Pixel size in axial direction in nm |
| `pixel_size_lateral_nm` | float | `100.0` | Pixel size in lateral direction in nm |
| `OVER_SAMPLING` | float | `4.0` | Oversampling factor for PSF computation |
| `accuracy` | int | `32` | Numerical integration accuracy |

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses the CLI11 library for command-line argument parsing.
- The `nlohmann/json` library is used for reading and handling JSON files.
- `FFTW` is used for fast Fourier transformations during the deconvolution process.
- `cuFFT` is used for GPU-accelerated FFT operations when CUDA is enabled.
- `OpenMP` is used for parallel FFTW threading.
- `ITK` is used for image processing.
- `spdlog` is used for logging.
- Icon attribution <a href="https://www.flaticon.com/free-icons/whale" title="whale icons">Whale icons created by Freepik - Flaticon</a>

---

## Contact

For questions or feedback, please contact [christoph.manitz@uni-jena.de].

---

