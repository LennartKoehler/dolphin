<div style="display: flex; align-items: center;">
    <img src="icon.png" alt="Whale Icon" width="60" height="60" style="margin-right: 10px;">
    <h1>DeconvTool v1.5.0</h1>
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
- **Support for a second PSF**: Users can provide or generate a secondary PSF for specific layers or cubes.
- **Flexible Parameters**: Adjustable parameters such as sigma values for synthetic PSF generation, iteration counts, lambda for regularization, and more.
- **Image Subdivision**: Processes images as grids of smaller cubes for memory efficiency and better performance.
- **Time Measurement**: Option to display the duration of deconvolution processes.
- **Configuration via CLI or JSON**: Users can specify parameters through command-line arguments or by providing a JSON configuration file.

## Requirements

- C++17 or later
- [OpenCV](https://opencv.org/) (for image processing)
- [FFTW](http://www.fftw.org/) (for fast Fourier transforms)
- [CLI11](https://github.com/CLIUtils/CLI11) (for command-line parsing)
- [nlohmann/json](https://github.com/nlohmann/json) (for JSON handling)

## Usage

### Command-Line Options

DeconvTool provides a variety of command-line options:

```
-i, --image <path>               Input Image Path (required)
-p, --psf <path>                 Input PSF Path or 'synthetic' (required)
--psf2 <path>                    Input second PSF Path or 'synthetic'
-a, --algorithm <algorithm>      Algorithm Selection ('rl'/'rltv'/'rif'/'inverse') (required)
--dataFormatImage <format>       Data Format for Image ('FILE'/'DIR') (required)
--dataFormatPSF <format>         Data Format for PSF ('FILE'/'DIR') (required)
--psfmodel <model>               PSF Model Selection ('gauss', ...)
--sigmax <value>                 SigmaX for synthetic PSF [5]
--sigmay <value>                 SigmaY for synthetic PSF [5] 
--sigmaz <value>                 SigmaZ for synthetic PSF [5] 
--psfx <value>                   Width of synthetic PSF [20]
--psfy <value>                   Height of synthetic PSF [20]
--psfz <value>                   Depth of synthetic PSF [30]
--iterations <value>             Number of iterations [10] (for RL)
--lambda <value>                 Lambda for Regularized Inverse Filter [1e-2]
--epsilon <value>                Epsilon for complex division [1e-6]
--borderType <type>              Border type for image extension [2] (0=constant, 1=replicate, 2=reflect)
--psfSafetyBorder <value>        Padding around the PSF [10]
--cubeSize <value>               Edge length of grid cubes [0]
--sigmax_2 <value>               SigmaX for the second synthetic PSF [10]
--sigmay_2 <value>               SigmaY for the second synthetic PSF [10]
--sigmaz_2 <value>               SigmaZ for the second synthetic PSF [15]
--savepsf                        Save the PSF used in the process
--time                           Show the processing time
--grid                           Process image in sub-images/cubes (grid)
--secondpsflayers <list>         list of cube layers for the second PSF
--secondpsfcubes <list>          list of cubes for the second PSF
--seperate                       Save image layers separately
--info                           Print information about the input image
--showExampleLayers              Display example layers of the image and PSF
-c, --config <path>              Path to JSON configuration file (required if no CLI arguments are provided)
```

### Example

```bash
./deconvtool -i input_image.tif -p synthetic -a rl --dataFormatImage FILE --dataFormatPSF FILE --sigmax 20 --sigmay 20 --sigmaz 20 --iterations 200 --time
```

This command will run the Richardson-Lucy algorithm with a synthetic PSF and specified sigma values, using the input image file and displaying the time taken.

### Using a Configuration File

You can specify your input, PSF, and other parameters using a JSON file. An example of the JSON configuration file is shown below:

```json
{
    "image_path": "input_image.tif",
    "psf_path": "synthetic",
    "algorithm": "rl",
    "dataFormatImage": "FILE",
    "dataFormatPSF": "FILE",
    "sigmax": 25.0,
    "sigmay": 25.0,
    "sigmaz": 25.0,
    "iterations": 100,
    "lambda": 1e-20,
    "epsilon": 1e-12,
    "psfSafetyBorder": 20,
    "cubeSize": 50,
    "time": true,
    "grid": true,
    "info": true,
    "secondpsflayers": [2, 4, 6],
    "secondpsfcubes": [1, 3, 5]
}
```

You can run the tool using the configuration file like this:

```bash
./deconvtool -c config.json
```

### Output

The processed images are saved in the specified format, and optional PSF files can be saved if the `--savepsf` flag is set. Additional information, such as the time taken for processing, will be displayed if the `--time` option is enabled.

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

**Note**: Replace placeholders like `<path_to_image>`, `<algorithm>`, etc., with actual values when running the tool.

