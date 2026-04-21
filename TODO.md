make the padding size not be the entire psf, but rather set a thershold for psf and then just pad the size of the psf where each value within the psf is larger than that threshold. This basically means only pad up until the psf fades off and, the small values dont have that much of an influence

could have different deconvolutionconfigs for different algorithms, similar to psfconfig, but also kinda unnecessary

when computing theoretical psf for deconvolution. Just directly compute it for the size of the image that will be used (padded image) instead of manually saying what size and then just zero padding to the size of the image

make seperate padding value for inside the image vs outside

check all the core math operations like addtoone if they correctly use stride

in place fft seems to work, but need to check if the result is aactually correct

can i somehow get around creating an out of place fft plan while still being faster, this plan init takes long and takes memory

the nicest thing to work with all the different data allocations for fft etc would be to have the managedData have a operations vector attached to it, and then whenever i call a function on that data it would not execute but just add it to the operations. And then i have like an execute function on the data which runs all the operations in order. Then before execute i can call "allocate" and it will allocate depending on if i will need to pad it for fft etc. So that all operations are known before actually running them

something in the memory layout and in place fft and the normal compute like division doesnt work. I assume some values are being used in math that arent supposed to be used and therefoe create weird values, all fft need to currently be in place. Also make the shape converion in cpubackend from and to real and complex nicer, i think there should only be once place where this is determined, check the next couple entries in todo

can cant use data with padding (for inplace) for out of place fft, because i then give the dimension and fftw doesnt know that ever 10th value in memory is actually for the padding, out of place fft expects contiguous memory, which the inplace memory is not. I could use the advanced plan creation, there these strides etc can be specified

something with in place forward fft r2c doesnt work, perhaps because i got the dimension ordering wrong or there is some other data getting into the buffer

use plans in place so that i dont need both plans, currently i need to save both out of place an inplace seperately, the manual says it has to be like this

in and out of place for cuda

only be able to reinerpret data that was allocated with buffer for other type

the memory management for the backend is now all over the place, now that im using real valuues and complex values and can change between them. Perhaps i need a interface and have backendspecific implementations

something with data view and memory allocation of real valued vs complex doesnt work. Perhaps its trying to free more than it allocated or the other way around

for example in rltv, it would be smart to compute total variation before allocating for the deconvolution task because then if i have a memory max, i dont have everything allocated at once

r2c fft in place and memory efficient

deconvolution of two rectangles doesnt work as expected

when using gpu, so lower memory and image doesnt fit on gpu with padding then the imagesplit recursion will run infinitely

with new implementation of no padding around the image, padding to the next fastest size of fftw now creates a different result, is this ok?

refactoring the split image to not have imagepadding on the outside:
    see how the ideal cube size is computed and see how the image is split into cubes. I dont want padding outside of the original image as its unnecessary.
    so if the image is larger than the psf then it should never be bigger (except to get a good computational size for fftw like multiple of 2)
    but there should only be padding between cubes in the inside of the image, never on the outside, like it was the case before.
    so if i dont adjust the size for a good fftw size then the pad image function should never be called in the tiffreader. only ever do i need padding between cubes, which is no padding but actually just a part of the image considered padding, but not artificial padding

i dont necessarily have to pad the whole image, cubes should be padded, but the padding around the image is not necessary as its artificial anyway, so might aswell just take cyclic boundary instead of reflecting

octant fourier shift for uneven dimensions (not divisible by 2)

check the testalgorithm for cuda there is a line on the left, seems like some misalignement
for cpu its even weirder, something is still wrong

i tihnk all the backdchecks can be turned into asserts as that should never happen except for out of memory

maybe make a filter for padding, so that i dont pad the entire psf, but just the part of the psf that has values over some threshold, then large psfs which mostly have very small numbers will have less of a padding and thereforecompute overhead

add threads to deconvolutionservice and psfservice

still not the same result as using only complex data, keep investigating

the fftw plans can be saved to file to not always recompute, however they are probably fixed to a specific ft size, check how saving this wisdom can be used to not always re init the plan
maybe this can be combined with finding a nice image size. So create discrete plans for good shapes and then smaller images are padded to that shape. Thne only save a coupleplans

i think i should also fix the complex octant fourier shift as its also probably wrong. chaning the real fourier shift solve the problem for me

testalgorithm with just backward transforming might now work because the padding is cut off and then nothing of the psf is left as its padded

in the full compelx backwardfft the normalization is now off because the shape of the data is lost, need to rethink how i manage the new complex data dimensions dimensions

warning currently its hardcoded in ibackendmemorymanager that the complex type is is half the "size" as the real type, this might not be true for all backends, perhaps this function needs to be moved back to the individual implementations

since i now have real_valued_data the transfer from and to itk image is probably easire, i tihnk i should somewhere be able to set a datapointer and thats it, keep the data

check how the real to complex and c2r work and how the complex data is then structured, because its smaller than the full complex data and if the steps afterwards have to be adjusted for this data, currently the image is nans

When finding uinque labels per cube also create the mask in the same loop
    - start creating a mask for the first two labels, the background label and the first label that shows.
    - Then when a new unique label is found just copy the first mask and "invert it" or just reinterpret later

FIX: make the fourier transforms templated for different datatypes, currently the real valued planinit is never used, thats the bug

check out fft r2c in place

use realdata for other deconvalgorithms

check which other operations need a realvalued counterpart
like total variation

make new value type for complexdata that is missing second half
probably dont need octant fouerier shift?

make cuda copy operations in the cudamemorybackend not the CUBE library

make make real value and template the backendmemorymanager to be able to take complex and realvalue, each should have a function which passes its size
or think about having a baseclass and then these two implementations, but then copying becomes weird. But i need real value for r2c and c2r fourier transforms which should be a big speedup
think more about complex/real valus in deconvolution

recheck the blocking behavior of using openmp

first forward and last backward fourier transforms can be real to complex

backwardfft in cuda divide by number of elementds


license -> under gnu, not mit? ->psf generator code

check if labeled deconvolution actually works, i think sinnme the values of the differents psfs are very different they basically get overwritten?

when do i normalize to with 1/N? after every IFFT or just at the end of the computation

does deconvnolutionprocessor need to own the algorithm?

threadpool destructor sometimes fails to join all threads. idk why, im guessing they might be stuck somewhere or lost? -> have logging but doesnt seem to happen anymore

if a mask is provided in labeleddeconvolution, but never used, if this is the largest mask, then it will still pad to this, is this fine?
    -> every cube will be padded to the largest psf provided in the config, doesnt matter if this psf is not used in this cube or if this psf is not used at all
    -> but this guarantees that there is only ever one fftw plan created

make background mask for the original image, so if there are some regions where no psfs are used the original image is just pasted there
feathering: last mask can not be seen as the background, because if there is a section where 3 masks meet, the 2 earlier masks would have weight ~0.5 and therefore the last mask would get 1 - 0.5 - 0.5 = 0. It should however be 1/3 for all. Only if there are only 2 masks can i do this.

make some operations doable in the image (convert image to complexdata)transofrmation function which does pixelwise operation anyway. When looping over each pixel, may aswell already do an operation on it, both directions. Perhaps make the conversion part of the Image3D object and then directly access a vector of functions?

distance map can be computed before deconvolution, then i can just copy in the pixel value multiplied by its weight

Danielssondistancemapimagefilter very slow in labeled image deconvolution, takes 50% of the total runtime, apparently even in release mode, check if itk is actually in release mode

include fft plans in the memory calculattion

if the setup is extremely cpu memory bound then using multiple openmp devices might be useful, then it uses less memory but one can still have two threads working on a task

getDevicememory in getMaxMemorypercube bugged for cudadevice, idk why

the memory transfer between host and device can be improved, as all tasks have same memory size
cna i somehow make the memory used for the workerdevice in a fixed position and pinned, e.g. for cuda i can always reuse the same location to load data into (or 2 locations if 2 iothreads). Is there a speedup of memory reading writing that this would get?

split library in public and private: currently the include paths are wrong after installation, i want the installed headers inside the library named directory in /usr/local/include/dolphin but then when using the code in the frontend dolphin doesnt find its own libraries because it looks in /usr/local/include, not in .../dolphin so private headers should stay as is while public headers i need to use the namespaced include <dolphin/PSFConfig.h> etc


make check that feathering radius is smaller than padding (psf size) so that there are no boundary conditions caused by too much feathering



Reader/Writer:
    should the reader actually have a lot of padding logic? i think no but its much easier because it has all the dimensionality information and its easir to pass as padding something that goes out of bounds of image instead of passing e.g. as negative values. Also it can load data however it sees fit. i dont think the executor should know about imagemetadata and determine itself what part of the image can be read and what has to padded after having read the image. I still believe that the reader should not do padding, but think about this before implementing


some weird padding problems, causing sufficient padding to still produce boundary conditions
rdata is read or written as when i init the result as zero the whole image is filled,
    for some reason half of the psf on both sides id not anough although mathematically it should be
    should also have nothing to do with normalization within the cube as i also tested that




- add threads and memory restrictions to psfgenerator and the corresponding conifg

- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1

- check if the deconvolutionalgorithms are correct







questions to ask:

- where should memory handling be?

- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




