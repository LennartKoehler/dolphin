currently with the changes ot the backendconfig, as this already includes the device, the createbackendiwthsharedmemory is not really needed, as that just used the backends config to configure, now i can just manually take that backends config. Feels like two systems doing the same job

in the backend properly write the move to device and copy to device aswell as move from and copy from, use move semantics and then make the input nullptr

check that psf ids are unique, for the supplied labelmap and for the used psf objects, and that theyre not overlapping

check everything including padding etc for uneven sizes

recheck the blocking behavior of using openmp

make some operations doable in the image (convert image to complexdata)transofrmation function which does pixelwise operation anyway. When looping over each pixel, may aswell already do an operation on it, both directions. Perhaps make the conversion part of the Image3D object and then directly access a vector of functions?

distance map can be computed before deconvolution, then i can just copy in the pixel value multiplied by its weight

Danielssondistancemapimagefilter very slow in labeled image deconvolution, takes 50% of the total runtime, apparently even in release mode, check if itk is actually in release mode

the memory transfer between host and device can be improved, as all tasks have same memory size
cna i somehow make the memory used for the workerdevice in a fixed position and pinned, e.g. for cuda i can always reuse the same location to load data into (or 2 locations if 2 iothreads). Is there a speedup of memory reading writing that this would get?

could make a low memory version for richardsonlucydeconvolution where the f is recomputed in backwardfft so that it doesnt have to be stored



plans for future:


make fusable compute kernels for the backend, so that not every function iterates over the data, but rather the backend stores a vector of operations (similar to the operations in image3d) and then only iterates data once. This makes data access much fasted. Then in the code that uses the backend each function can either be fused or run instantly. This allows the user to either build pipelines (if its the same data) or just run function as is (e.g. for fourier transform the data acess is irregular anyway) but if multiple functions that access data and are elementwise are run after oneanother then this can be used. E.g. for the aberration i can then have a fused kernel that adds the phase and then in the same kernel computed the error.
