make check that labeled image is same size as deconvolve image

check that psf ids are unique, for the supplied labelmap and for the used psf objects, and that theyre not overlapping

for the labeleddeconvolution, currently when iterating the labeled image the labelimagerange is not taken into account, so these ranges currently have no influence, but this can be done in the hot loop, to  create the masks right away. Just see if the labelvalue is in the range specified and not just for unique labels. Currently it just checks how many uniquelabels exist instead of seeing if its withing a specified range

should the tiffreader hold the image memory

maxmemory management

check everything including padding etc for uneven sizes

check all the core math operations like addtoone if they correctly use stride

in place fft seems to work, but need to check if the result is aactually correct

can i somehow get around creating an out of place fft plan while still being faster, this plan init takes long and takes memory

when using gpu, so lower memory and image doesnt fit on gpu with padding then the imagesplit recursion will run infinitely

octant fourier shift for uneven dimensions (not divisible by 2)

recheck the blocking behavior of using openmp

When finding uinque labels per cube also create the mask in the same loop
    - start creating a mask for the first two labels, the background label and the first label that shows.
    - Then when a new unique label is found just copy the first mask and "invert it" or just reinterpret later


check if labeled deconvolution actually works, i think sinnme the values of the differents psfs are very different they basically get overwritten?

if a mask is provided in labeleddeconvolution, but never used, if this is the largest mask, then it will still pad to this, is this fine?
    -> every cube will be padded to the largest psf provided in the config, doesnt matter if this psf is not used in this cube or if this psf is not used at all
    -> but this guarantees that there is only ever one fftw plan created

make background mask for the original image, so if there are some regions where no psfs are used the original image is just pasted there
feathering: last mask can not be seen as the background, because if there is a section where 3 masks meet, the 2 earlier masks would have weight ~0.5 and therefore the last mask would get 1 - 0.5 - 0.5 = 0. It should however be 1/3 for all. Only if there are only 2 masks can i do this.

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




