discuess about how much padding is needed for the different padding types. Given a psf how do i determine how much to pad, how much padding should be between the cubes? full padding or also less?

check that psf ids are unique, for the supplied labelmap and for the used psf objects, and that theyre not overlapping

should the tiffreader hold the image memory

maxmemory management

check everything including padding etc for uneven sizes

check all the core math operations like addtoone if they correctly use stride

when using gpu, so lower memory and image doesnt fit on gpu with padding then the imagesplit recursion will run infinitely

octant fourier shift for uneven dimensions (not divisible by 2)
also apparently doesnt correctly use strided input

the sumToOne uses raw pointer, this has no chance of using strides, if all are same stride then ok, but could be nicer

recheck the blocking behavior of using openmp

make some operations doable in the image (convert image to complexdata)transofrmation function which does pixelwise operation anyway. When looping over each pixel, may aswell already do an operation on it, both directions. Perhaps make the conversion part of the Image3D object and then directly access a vector of functions?

distance map can be computed before deconvolution, then i can just copy in the pixel value multiplied by its weight

Danielssondistancemapimagefilter very slow in labeled image deconvolution, takes 50% of the total runtime, apparently even in release mode, check if itk is actually in release mode

include fft plans in the memory calculattion

the memory transfer between host and device can be improved, as all tasks have same memory size
cna i somehow make the memory used for the workerdevice in a fixed position and pinned, e.g. for cuda i can always reuse the same location to load data into (or 2 locations if 2 iothreads). Is there a speedup of memory reading writing that this would get?

make check that feathering radius is smaller than padding (psf size) so that there are no boundary conditions caused by too much feathering



Reader/Writer:
    should the reader actually have a lot of padding logic? i think no but its much easier because it has all the dimensionality information and its easir to pass as padding something that goes out of bounds of image instead of passing e.g. as negative values. Also it can load data however it sees fit. i dont think the executor should know about imagemetadata and determine itself what part of the image can be read and what has to padded after having read the image. I still believe that the reader should not do padding, but think about this before implementing







- check if the deconvolutionalgorithms are correct

questions to ask:

- where should memory handling be?

- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




