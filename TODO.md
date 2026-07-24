currently with the changes ot the backendconfig, as this already includes the device, the createbackendiwthsharedmemory is not really needed, as that just used the backends config to configure, now i can just manually take that backends config. Feels like two systems doing the same job

crashes if memory in config is not set propery? should not happen

there might still be wrong with how the subimages are stitched back together, there is an artifact when using "parent" padding strategy althugh this should be sufficient padding

error when psf config or other path not found


in the backend properly write the move to device and copy to device aswell as move from and copy from, use move semantics and then make the input nullptr

make some of the padding variations user interactive, like if the psf that is read from file is larger than the cube that would be processed (due to memory) then prompt the user to specify if either the psf is cut off or the cube is enlarged.

standarddeconvstrategy, creating psfs needs the threadpool and progresscallback

gibsonlanni psf generator, how much needs to be padded?

seperate image padding and cubepadding? should be same or not?

discuess about how much padding is needed for the different padding types. Given a psf how do i determine how much to pad, how much padding should be between the cubes? full padding or also less?

check that psf ids are unique, for the supplied labelmap and for the used psf objects, and that theyre not overlapping

should the tiffreader hold the image memory

maxmemory management

check everything including padding etc for uneven sizes

check all the core math operations like addtoone if they correctly use stride

octant fourier shift for uneven dimensions (not divisible by 2)
also apparently doesnt correctly use strided input

the sumToOne uses raw pointer, this has no chance of using strides, if all are same stride then ok, but could be nicer

recheck the blocking behavior of using openmp

make some operations doable in the image (convert image to complexdata)transofrmation function which does pixelwise operation anyway. When looping over each pixel, may aswell already do an operation on it, both directions. Perhaps make the conversion part of the Image3D object and then directly access a vector of functions?

distance map can be computed before deconvolution, then i can just copy in the pixel value multiplied by its weight

Danielssondistancemapimagefilter very slow in labeled image deconvolution, takes 50% of the total runtime, apparently even in release mode, check if itk is actually in release mode

the memory transfer between host and device can be improved, as all tasks have same memory size
cna i somehow make the memory used for the workerdevice in a fixed position and pinned, e.g. for cuda i can always reuse the same location to load data into (or 2 locations if 2 iothreads). Is there a speedup of memory reading writing that this would get?

could make a low memory version for richardsonlucydeconvolution where the f is recomputed in backwardfft so that it doesnt have to be stored






- check if the deconvolutionalgorithms are correct

questions to ask:

- where should memory handling be?

- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




plans for future:


make fusable compute kernels for the backend, so that not every function iterates over the data, but rather the backend stores a vector of operations (similar to the operations in image3d) and then only iterates data once. This makes data access much fasted. Then in the code that uses the backend each function can either be fused or run instantly. This allows the user to either build pipelines (if its the same data) or just run function as is (e.g. for fourier transform the data acess is irregular anyway) but if multiple functions that access data and are elementwise are run after oneanother then this can be used. E.g. for the aberration i can then have a fused kernel that adds the phase and then in the same kernel computed the error.
