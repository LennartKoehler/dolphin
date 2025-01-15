<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>DeconvTool v1.6.1</h1>
</div>


---

DeconvTool is a C++ command-line tool designed for deconvolution of microscopy images. It supports multiple deconvolution algorithms and allows the generation and use of synthetic Point Spread Functions (PSF). The tool is intended for users familiar with image processing and deconvolution techniques.

## Features

- **Input Image Formats**: Supports both single image files and directories of image slices in TIF format.
- **Point Spread Function (PSF) Input**: Allows users to provide a PSF as a file, a directory of slices, or generate a synthetic PSF.
- **Multiple Deconvolution Algorithms**:
    - Richardson-Lucy (RL)
    - Richardson-Lucy with Total Variation (RLTV)
    - Regularized Inverse Filter (RIF)
    - Inverse Filter
- **Support for multiple PSFs**: Users can provide or generate multiple PSFs for specific layers or subimages.
- **Flexible Parameters**: Adjustable parameters such as sigma values for synthetic PSF generation, iteration counts, lambda for regularization, and more.
- **Image Subdivision**: Processes images as grids of smaller subimages for memory efficiency and better performance.
- **Time Measurement**: Option to display the duration of deconvolution processes.
- **Configuration via CLI or JSON**: Users can specify parameters through command-line arguments or by providing a JSON configuration file.

## Requirements

- C++17 or later
- [OpenCV](https://opencv.org/) (for image processing)
- [FFTW](http://www.fftw.org/) (for fast Fourier transforms)
- [CLI11](https://github.com/CLIUtils/CLI11) (for command-line parsing)
- [nlohmann/json](https://github.com/nlohmann/json) (for JSON handling)

## Build
First build the CUBE library in /lib/cube/.
```bash
cd ./lib/cube/
mkdir build
cd build

cmake ..
make
```
Then the DeconvTool:
```bash
mkdir build
cd build

cmake ..
make
```

## Usage

### Command-Line Options

DeconvTool provides a variety of command-line options:

```
-i, --image <path>               Input Image Path (required)
-p, --psf <path>                 Input PSF Path (required)
-a, --algorithm <algorithm>      Algorithm Selection ('rl'/'rltv'/'rif'/'inverse') (required)
--iterations <value>             Number of iterations [10] (for RL)
--lambda <value>                 Lambda for Regularized Inverse Filter [1e-2]
--epsilon <value>                Epsilon for complex division [1e-6]
--borderType <type>              Border type for image extension (0=constant, 1=replicate, 2=reflect) [2] 
--psfSafetyBorder <value>        Padding around the PSF [10] (pixel)
--subimageSize <value>           Edge length of grid subimages [0] (pixel, 0 = auto adjust to PSF)
--gpu <type>                     Type of GPU API ('cuda'/'none') [none]
-c, --config <path>              Path to JSON configuration file (required if no CLI arguments are provided)
Flags:
--savepsf                        Save the PSF used in the process [false]
--time                           Show the processing time [false]
--grid                           Process image in subimages [false]
--seperate                       Save image layers separately [false]
--info                           Print information about the input image [false]
--showExampleLayers              Display example layers of the image and PSF [false]
```

The PSF can be provided as a TIF file, a TIF directory (where each layer is a separate TIF file), or a JSON configuration file. The JSON configuration file can be used to generate a synthetic PSF or specify the path to a file or directory. With a configuration-based PSF file, it is possible to target specific sub-images for processing. This requires an array of PSFs as input. Only the first PSF in the array will be processed globally, while all subsequent PSFs will be processed based on their corresponding sub-image and layer IDs. If no position is specified, the PSF will not be applied. If a position is provided multiple times, only the PSF with the smaller index in the array will be applied; subsequent PSFs at the same position will be ignored. The following JSON configuration show a example configuration of a synthetic PSF and a configuration file with a file path.
```
{
  "sigmax": 2.5,
  "sigmay": 2.5,
  "sigmaz": 2.5,
  "psfx": 20,
  "psfy": 20,
  "psfz": 46,
  "psfmodel": "gauss",
  "layers": [2,3],
  "subimages": [136]
}
```
```
{
  "path": "../input/psf.tif",
  "layers": [3,4],
  "subimages": [10,11,12,16]
}
```
DeconvTool provides this parameters for a PSF configuration:
```
path                             Path to PSF TIF file or dir (string "" or json array of strings["",""])
psfx                             X dimension of PSF (pixel) (integer)
psfy                             Y dimension of PSF (pixel) (integer)
psfz                             Z dimension of PSF (pixel) (integer)
psfmodel                         PSF model for generation ('gauss')
sigmax                           Spread of data in x dim, the larger σ, the wider and flatter the bell curve (double)
sigmay                           Spread of data in y dim, the larger σ, the wider and flatter the bell curve (double)
sigmaz                           Spread of data in z dim, the larger σ, the wider and flatter the bell curve (double)
layers                           Specific layers, where PSF will be applied (json array of integers [])
subimages                        Specific subimages, where PSF will be applied (json array of integers [])
```
### Example

```bash
./deconvtool -i input_image.tif -p psf.tif -a rl --iterations 100 --time
```

This command will run the Richardson-Lucy algorithm with a PSF file using the input image file and displaying the time taken.

```bash
./deconvtool -i input_image.tif -p psf_syn.json psf_path.json -a rltv --iterations 50 --info
```

This command will run the Richardson-Lucy with Total Variation algorithm with a synthetic PSF using a config file globally and a PSF file locally through antoher config file. The metadata of the input image will displayed. 
### Using a Configuration File

You can specify your input, PSF, and other parameters using a JSON file. An example of the JSON configuration file is shown below:

```json
{
  "image_path": "../input/your_image.tif",
  "info": false,
  "showExampleLayers": false,
  "time": false,
  "seperate": false,

  "algorithm": "rltv",
  "epsilon": 1e-6,
  "lambda": 0.015,
  "iterations": 10,

  "psf_path": ["../configs/default_psf.json"],
  "savePsf": false,

  "grid": true,
  "borderType": 2,
  "cubeSize": 0,
  "psfSafetyBorder": 10,

  "gpu": "none"
}
```
DeconvTool creates two executable if CUDA is availabe on your system: deconvtool and deconvtoolcuda. Make sure your specify the "gpu" parameter with "cuda" if your are using the GPU accelerated version deconvtoolcuda.
You can run the tool using the configuration file like this:

```bash
./deconvtool -c config.json
```
```bash
./deconvtoolcuda -c config_gpu.json
```

### Output

The processed images are saved in the TIF format, if the `--seperate` flag is set the image will also saved as a directory where every layer is a single TIF file, and optional PSF files can be saved if the `--savepsf` flag is set. Additional information, such as the time taken for processing, will be displayed if the `--time` option is enabled.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The project uses the CLI11 library for command-line argument parsing.
- The `nlohmann/json` library is used for reading and handling JSON files.
- The `OpenCV` library facilitates image processing tasks.
- `FFTW` is used for fast Fourier transformations during the deconvolution process.
- Icon attribution <a href="https://www.flaticon.com/free-icons/whale" title="whale icons">Whale icons created by Freepik - Flaticon</a>

---

## Contact

For questions or feedback, please contact [christoph.manitz@uni-jena.de].

---

