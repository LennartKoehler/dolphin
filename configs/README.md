# Configuration Parameters

## Required Parameters
- `-i, --image <path>`
- `-p, --psf <path>`
- `-a, --algorithm <algorithm>`
- `--dataFormatImage <format>`
- `--dataFormatPSF <format>`

## Input and Output

- `-i, --image <path>`  
  **Description**: Path to the input image file.  
  **Required**: Yes.

- `-p, --psf <path>`  
  **Description**: Path to the input PSF file or use `'synthetic'` to generate a synthetic PSF.  
  **Required**: Yes.

- `--psf2 <path>`  
  **Description**: Path to the second input PSF file or use `'synthetic'` to generate a second synthetic PSF.  
  **Required**: No.

- `--savepsf`  
  **Description**: Save the PSF used in the process.  
  **Default**: 'false'.  
  **Required**: No.

- `--info`  
  **Description**: Print information about the input image.  
  **Default**: 'false'.  
  **Required**: No.

- `--showExampleLayers`  
  **Description**: Display example layers of the image and PSF.  
  **Default**: 'false'.  
  **Required**: No.

## Algorithm Selection

- `-a, --algorithm <algorithm>`  
  **Description**: Select the algorithm to be used. Options are:
    - `'rl'`: Richardson-Lucy Deconvolution.
    - `'rltv'`: Richardson-Lucy Total Variation.
    - `'rif'`: Regularized Inverse Filter.
    - `'inverse'`: Inverse filtering.  
      **Required**: Yes.

## Data Format

- `--dataFormatImage <format>`  
  **Description**: Specifies the data format for the input image. Options are:
    - `'FILE'`: Single image file.
    - `'DIR'`: Directory containing image slices.  
      **Required**: Yes.

- `--dataFormatPSF <format>`  
  **Description**: Specifies the data format for the input PSF. Options are:
    - `'FILE'`: Single PSF file.
    - `'DIR'`: Directory containing PSF slices.  
      **Required**: Yes.

## Synthetic PSF Parameters

- `--psfmodel <model>`  
  **Description**: Select the PSF model to be used. Options are:
    - `'gauss'`: Gaussian PSF.
    - `'bornwolf'`: Born-Wolf PSF (coming soon).
  
  **Default**: 'gauss'.  
  **Required**: No.

- `--sigmax <value>`  
  **Description**: Sigma value in the X direction for the synthetic PSF.  
  **Default**: 5.  
  **Required**: No.

- `--sigmay <value>`  
  **Description**: Sigma value in the Y direction for the synthetic PSF.  
  **Default**: 5.  
  **Required**: No.

- `--sigmaz <value>`  
  **Description**: Sigma value in the Z direction for the synthetic PSF.  
  **Default**: 5.  
  **Required**: No.

- `--psfx <value>`  
  **Description**: Width in px of the synthetic PSF.  
  **Default**: 20.  
  **Required**: No.

- `--psfy <value>`  
  **Description**: Height in px of the synthetic PSF.  
  **Default**: 20.  
  **Required**: No.

- `--psfz <value>`  
  **Description**: Depth in px of the synthetic PSF.  
  **Default**: 30.  
  **Required**: No.

- `--sigmax_2 <value>`  
  **Description**: Sigma value in the X direction for the second synthetic PSF.  
  **Default**: 10.  
  **Required**: No.

- `--sigmay_2 <value>`  
  **Description**: Sigma value in the Y direction for the second synthetic PSF.  
  **Default**: 10.  
  **Required**: No.

- `--sigmaz_2 <value>`  
  **Description**: Sigma value in the Z direction for the second synthetic PSF.  
  **Default**: 15.  
  **Required**: No.

## Deconvolution and Regularization

- `--iterations <value>`  
  **Description**: Number of iterations for the Richardson-Lucy deconvolution algorithm.  
  **Default**: 10.  
  **Required**: No.

- `--lambda <value>`  
  **Description**: Lambda parameter for the Regularized Inverse Filter.  
  **Default**: 1e-2.  
  **Required**: No.

- `--epsilon <value>`  
  **Description**: Epsilon value for avoiding division by zero in complex division.  
  **Default**: 1e-6.  
  **Required**: No.

## Image Processing and Grid Parameters

- `--borderType <type>`  
  **Description**: Specifies the border type for image extension. Options are:
    - `0`: Constant border. (constant 0)
    - `1`: Replicate border. (constant last value)
    - `2`: Reflect border. (mirrored image)
      **Default**: 2.  
      **Required**: No.

- `--psfSafetyBorder <value>`  
  **Description**: Padding around the PSF to prevent edge effects.  
  **Default**: 10.  
  **Required**: No.

- `--cubeSize <value>`  
  **Description**: Edge length in px of the grid cubes used for processing.  
    - `0`: Automatic fiiting to PSF size.
    
  **Default**: 0.  
        **Required**: No.

- `--grid`  
  **Description**: Process the image in sub-images (grid mode).  
  **Default**: 'false'.  
  **Required**: No.

- `--secondpsflayers <list>`  
  **Description**: A list of sub-image/cube layers for the second PSF, which can be specified for processing.  
  **Default**: `[]` (empty list)  
  **Required**: No.

- `--secondpsfcubes <list>`  
  **Description**: A list of cubes for the second PSF, which can be specified for processing.  
  **Default**: `[]` (empty list)  
  **Required**: No.

- `--separate`  
  **Description**: Save image layers separately after processing.  
  **Default**: 'false'.  
  **Required**: No.

## Configuration and Timing

- `-c, --config <path>`  
  **Description**: Path to the JSON configuration file. This parameter is required if no command-line arguments are provided.  
  **Required**: Yes, if no other CLI arguments are provided.

- `--time`  
  **Description**: Display the total processing time for the operation.
  **Default**: 'false'.  
  **Required**: No.
