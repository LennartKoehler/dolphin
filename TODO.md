- fix backend refactoring, cmake unfinished, how do backend implementatinos acces the complexdata implementation?

- connect grid processing to multithreading, so that number of owrker threads also dictates the gridsize, also perhaps include a maxMem flag

- add exception handling to threadpool, within the threadpool loop (which is defined in the constructor) each thread needs to handle any exceptions, like bad malloc. This has to be passed up the entire chain up until there starting from backend, only the backends should eveer be used for this. change the cvtofftw transformation to also use the cpubackend so this works. Then when this is triggered the thread that executed that task reschedules that task. The idea is that once a task is rescheduled there is no further task launched until a task finishes (-> which should free up memory), then perhaps all tasks continue as normal? think about this


- do backendmemorymanager and deconvolutionbackend need to share one backend mutex?
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





