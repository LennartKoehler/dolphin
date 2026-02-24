
threadpool destructor sometimes fails to join all threads. idk why, im guessing they might be stuck somewhere or lost?


adjust cuda backend

make background mask for the original image, so if there are some regions where no psfs are used the original image is just pasted there

openmp and cpubackend thread initialization, both init fftw, this is problematic, check how the thread init works et
work on the labeled image mask pipeline of using weighted masks

make some operations doable in the image (convert image to complexdata)transofrmation function which does pixelwise operation anyway. When looping over each pixel, may aswell already do an operation on it, both directions. Perhaps make the conversion part of the Image3D object and then directly access a vector of functions?

feathering: last mask can not be seen as the background, because if there is a section where 3 masks meet, the 2 earlier masks would have weight ~0.5 and therefore the last mask would get 1 - 0.5 - 0.5 = 0. It should however be 1/3 for all. Only if there are only 2 masks can i do this.

distance map can be computed before deconvolution, then i can just copy in the pixel value multiplied by its weight

make configuration for openmp backend so that the number of threads can be set. The problem is currently cpubackend is just a singleton and doesnt care. Where for cuda i have one backend per stream. But for openmp i need a mix. I want multiple openmp backends each with multiple threads

Danielssondistancemapimagefilter very slow in labeled image deconvolution, takes 50% of the total runtime, apparently even in release mode, check if itk is actually in release mode

include fft plans in the memory calculattion

if the setup is extremely cpu memory bound then using multiple openmp backends might be useful, then it uses less memory but one can still have two threads working on a task

getBackendmemory in getMaxMemorypercube bugged for cudabackend, idk why

the memory transfer between host and device can be improved, as all tasks have same memory size
cna i somehow make the memory used for the workerbackend in a fixed position and pinned, e.g. for cuda i can always reuse the same location to load data into (or 2 locations if 2 iothreads). Is there a speedup of memory reading writing that this would get?



make bordertypes in config into string


split library in public and private: currently the include paths are wrong after installation, i want the installed headers inside the library named directory in /usr/local/include/dolphin but then when using the code in the frontend dolphin doesnt find its own libraries because it looks in /usr/local/include, not in .../dolphin so private headers should stay as is while public headers i need to use the namespaced include <dolphin/PSFConfig.h> etc 

work on labeled deconvolution, make faster

make check that feathering radius is smaller than padding (psf size) so that there are no boundary conditions caused by too much feathering



Reader/Writer:
    should the reader actually have a lot of padding logic? i think no but its much easier because it has all the dimensionality information and its easir to pass as padding something that goes out of bounds of image instead of passing e.g. as negative values. Also it can load data however it sees fit. i dont think the executor should know about imagemetadata and determine itself what part of the image can be read and what has to padded after having read the image. I still believe that the reader should not do padding, but think about this before implementing


some weird padding problems, causing sufficient padding to still produce boundary conditions
rdata is read or written as when i init the result as zero the whole image is filled,
    for some reason half of the psf on both sides id not anough although mathematically it should be
    should also have nothing to do with normalization within the cube as i also tested that



think about how the rangemap for labelpsfmap is loaded. create a base loading function that will read from string, then create loading functions that will read from json obj, and or json file. think about how when and where the labelpsfmap should be loaded, i think it should be used in the deconvolutionstrategyfactory, and should also be loaded there and passed to the labeleddeconvolutionstrategy there, so that other strategies know nothing about it. on the other hand, should the factory really be doing loading? can i create a reader object that will handle all of that? perhaps one that can do tif reading, json reading etc. currently reading is all over the place, e.g. psfconfigs, where are those read? they "read themselves"



- add threads and memory restrictions to psfgenerator and the corresponding conifg

- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1

- check if the deconvolutionalgorithms are correct







questions to ask:

- where should memory handling be?
    
- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




