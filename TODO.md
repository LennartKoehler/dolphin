


- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1


- should backends be singletons?
    -> memory manager doesnt have a state anyway
    -> ideconvbackend yes, except if different fftw plans are needed(e.g. of different size)


- check if the deconvolutionalgorithms are correct






questions to ask:


- do we need a flag for padding dimensions -> with padding smaller than psf there would be grid artifacts due to cyclic deconv


- psfs for different regions of the image:
    How do we know which psf is for which region?
    Can all regions (cubes) still have the same size (for faster fft)? Can they still be cuboids?
    Think about the difference between using theoretical psfs vs measured psfs
    
- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




