- connect grid processing to multithreading, so that number of owrker threads also dictates the gridsize, also perhaps include a maxMem flag

- implement cubepsfmap, do we even need it?

- ask zoltan about image extension and padding, why should i pad the psf? makes sense that the cubes are padded
- manage the memory, currently it will hapily go out of memory on cpu and gpu without grid processing
- check if the deconvolutionalgorithms are correct








GUI:
a event system which lets lower layer windows be able to ocmmunicate with higher level would be nice






cubenumvec and layernumvec: what do they do
 LK these should allow differnt cubes and layers to be deconved with different psfs

secondarypsfcube / vec what do they do
 LK i think these are remains, replaced by cubenumvec to allow arbitrary number of psfs





