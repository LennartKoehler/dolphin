- check memory usage implementation
- maybe there should only be one threadpool, and everywhere new threads are started they simply enqueue a function and the available thread will then manage that
- implement cubepsfmap, do we even need it?

- make subimagesize a rectangleshape or make it if subimage size > originalimagesize to cap to that

- fftw plans could be passed around the backends, this seperates state from functionality

- ask zoltan about image extension and padding, why should i pad the psf? makes sense that the cubes are padded
- manage the memory, currently it will hapily go out of memory on cpu and gpu without grid processing
- check if the deconvolutionalgorithms are correct








GUI:
a event system which lets lower layer windows be able to ocmmunicate with higher level would be nice






cubenumvec and layernumvec: what do they do
 LK these should allow differnt cubes and layers to be deconved with different psfs

secondarypsfcube / vec what do they do
 LK i think these are remains, replaced by cubenumvec to allow arbitrary number of psfs





