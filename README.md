<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="50" height="50" style="margin-right: 10px;">
    <h1>DeconvTool</h1>
</div>

## Abstract

DeconvTool is an open-source command-line-based microscopy image deconvolution tool designed for organ-on-chip applications. Written in C++, DeconvTool enables users to process image data and Point Spread Functions (PSFs) via command-line arguments or JSON-formatted configuration files. It supports both real and synthetic PSFs and offers various parameters for image grid processing, deconvolution algorithms, and custom image extensions.

The tool implements several deconvolution algorithms, including:
- **Inverse Filter Deconvolution**
- **Regularized Inverse Filter Deconvolution**
- **Richardson-Lucy Algorithm**

It allows for parameter tuning such as the number of iterations. Despite its flexibility, DeconvTool is slower compared to other tools like Huygens and DeconvLab2 and may introduce artifacts in some scenarios. Future enhancements aim to optimize performance through process parallelization and GPU acceleration. Improvements to PSF generation are also planned.

## Features

- **Command-Line Interface (CLI)**: Input parameters via command line or JSON configuration.
- **Algorithm Support**: Inverse Filter, Regularized Inverse Filter, Richardson-Lucy, and convolution.
- **Synthetic PSF Generation**: Support for synthetic PSFs with adjustable parameters.
- **Image Grid Processing**: Option to split images into subimages (cubes) for processing.
- **Custom Extensions**: Options for border types and image extension methods.

## Installation

To use DeconvTool, ensure you have a C++ compiler and necessary libraries installed. You can build DeconvTool by running:

```bash
mkdir build
cd build
cmake ..
make
```

## Usage

You can run DeconvTool via the command line with various options. Below are some examples of how to use it.

### Command-Line Arguments

```
deconvtool --image <path_to_image> --psf <path_to_psf> --algorithm <algorithm> --dataFormatImage <FILE|DIR> --dataFormatPSF <FILE|DIR> [options]
```

### Required Arguments

- `-i, --image` : Path to the input image.
- `-p, --psf` : Path to the PSF or keyword `synthetic` to generate a synthetic PSF.
- `-a, --algorithm` : Algorithm to use for deconvolution (`inverse`, `rl`, `rif`).
- `--dataFormatImage` : Data format for image (`FILE` or `DIR`).
- `--dataFormatPSF` : Data format for PSF (`FILE` or `DIR`).

### Optional Arguments

- `--sigmax`, `--sigmay`, `--sigmaz`: Parameters for synthetic PSF (default: 25).
- `--psfx`, `--psfy`, `--psfz`: Dimensions for synthetic PSF (default: 500x500x20).
- `--epsilon`: Epsilon for complex division (default: 0).
- `--borderType`: Type of border extension (0-constant, 1-replicate, 2-reflect) (default: 2).
- `--psfSafetyBorder`: Safety border around the PSF in pixels (default: 20).
- `--cubeSize`: Size of subimages (cubes) for grid processing (default: 50).
- `--iterations`: Number of iterations for iterative algorithms (default: 100).
- `--lambda`: Regularization parameter for Regularized Inverse Filter (default: 1e-20).
- `--savepsf`: Save the PSF after deconvolution (default: false).
- `--time`: Show duration of the deconvolution process.
- `--grid`: Enable image splitting into subimages.
- `--separate`: Save channels separately (for RL PNG).
- `--info`: Print metadata of the input image.
- `--showExampleLayers`: Show a layer of the loaded image and PSF.

### Configuration File

You can also use a JSON configuration file to specify the parameters. The configuration file should include the following fields:

```json
{
  "image_path": "path_to_image",
  "psf_path": "path_to_psf",
  "algorithm": "algorithm",
  "dataFormatImage": "FILE|DIR",
  "dataFormatPSF": "FILE|DIR",
  "sigmax": 25.0,
  "sigmay": 25.0,
  "sigmaz": 25.0,
  "psfx": 500,
  "psfy": 500,
  "psfz": 20,
  "epsilon": 0,
  "iterations": 100,
  "lambda": 1e-20,
  "psfSafetyBorder": 20,
  "cubeSize": 50,
  "borderType": 2,
  "sep": false,
  "time": false,
  "savePsf": false,
  "showExampleLayers": false,
  "info": false,
  "grid": true
}
```

Run DeconvTool with the configuration file using:

```bash
deconvtool --config <path_to_config_file>
```

## Performance and Limitations

DeconvTool may exhibit slower performance compared to other deconvolution tools and may introduce artifacts in certain scenarios. Future work includes optimizing performance with parallelization and GPU acceleration, as well as improving PSF generation.

## License

DeconvTool is open-source software licensed under the [MIT License](LICENSE). 

## Contact

For questions or feedback, please contact [christoph.manitz@uni-jena.de].
Icon attribution <a href="https://www.flaticon.com/free-icons/whale" title="whale icons">Whale icons created by Freepik - Flaticon</a>
---

**Note**: Replace placeholders like `<path_to_image>`, `<algorithm>`, etc., with actual values when running the tool.

