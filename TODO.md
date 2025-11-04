


- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1


- should backends be singletons?
    -> memory manager doesnt have a state anyway
    -> ideconvbackend yes, except if different fftw plans are needed(e.g. of different size)


- check if the deconvolutionalgorithms are correct






coordinate based psfs:
    - make the deconvolution take 3d coordinates for a box (so xyz and whd) which correspond to the input image, a psf, and apply the psf to those coordinates of the image. This needs to save the coordinates to the result to know where it came from
    - the coordinates should be any size. And then later on i can decide to make them all one size to speed up fourier plans
    - the deconvolution then takes a map of coordinates to psf. Then it just goes through that map and applies the deconvolution with the corresponding psf to that part of the image which is described by the coordinates.
    - make the map be able to take multiple psfs, and then iterate ofver all
    - then make gui which lets you choose the boxes accordingly (start off by creating a equal sized 3d grid of the image of small boxes) the user then assigns a psf to each box
psudocode:
    map = { box(1,5,5, 10,10,10) : [psfA], box(1,15,5, 10,10,10): [psfA,psfB]} // box (x,y,z, w,h,d)
    for box, psflist in map:
        for psf in psflist:
            subimage = image[box]
            deconvolve(subimage, psf)



questions to ask:


- do we need a flag for padding dimensions -> with padding smaller than psf there would be grid artifacts due to cyclic convolution


- psfs for different regions of the image:
    How do we know which psf is for which region?
    Can all regions (cubes) still have the same size (for faster fft)? Can they still be cuboids?
    Think about the difference between using theoretical psfs vs measured psfs
    
- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




