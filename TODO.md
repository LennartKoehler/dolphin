

make check that feathering radius is smaller than padding (psf size) so that there are no boundary conditions caused by too much feathering

make backend into ABI? correctly seperate backend from application? If it stays dlopen, then there is no real use inregistering backends, but rather we can just select a backend library through cli. Then there is no need for someone to have backend libraries that he doesnt use, just somehow always include cpubackend, as the default, and as its needed for the deconvolutionexecutor
    - create an IBackend abi which also declares a complexdata interface
    - in the implementations (cuda and cpu) the complexdata can have a definition (could be same definition for all)



tiffwriter doesnt work with yeast image



Reader/Writer:
    should the reader actually have a lot of padding logic? i think no but its much easier because it has all the dimensionality information and its easir to pass as padding something that goes out of bounds of image instead of passing e.g. as negative values. Also it can load data however it sees fit. i dont think the executor should know about imagemetadata and determine itself what part of the image can be read and what has to padded after having read the image. I still believe that the reader should not do padding, but think about this before implementing


    Think about imagePadding, if i want to cut images out of the image, the deconvstrategy expects the originalImage to already be padded. How do i pad the image without having to load everything. The coordinates of how the cubes are requested are also for a padded image, think about this. Pad individual strips if they are at the edge of the image. Think about how coordinates are translated from unpadded to padded image psf padding is just zeros, so this doesnt actually matter. 



some weird padding problems, causing sufficient padding to still produce boundary conditions
    somehow caused by padding for whatever reason, should have nothing to do with how data is read or written as when i init the result as zero the whole image is filled,
    for some reason half of the psf on both sides id not anough although mathematically it should be
    should also have nothing to do with normalization within the cube as i also tested that



think about how the rangemap for labelpsfmap is loaded. create a base loading function that will read from string, then create loading functions that will read from json obj, and or json file. think about how when and where the labelpsfmap should be loaded, i think it should be used in the deconvolutionstrategyfactory, and should also be loaded there and passed to the labeleddeconvolutionstrategy there, so that other strategies know nothing about it. on the other hand, should the factory really be doing loading? can i create a reader object that will handle all of that? perhaps one that can do tif reading, json reading etc. currently reading is all over the place, e.g. psfconfigs, where are those read? they "read themselves"




- currently setting number threads to max will be slower than e.g. 8/12 because I assume the overhead is larger and there is no longer a benefit to having a 
- add threads and memory restrictions to psfgenerator and the corresponding conifg
- can i get around using the cpumemorymanager? somehow directly convert from cv mat to fftw?


- only difference between dolphins implementation of gibsonlanni and big psf generator is it doesnt scale the max to 1

- check if the deconvolutionalgorithms are correct







questions to ask:


- do we need a flag for padding dimensions -> with padding smaller than psf there would be grid artifacts due to cyclic convolution
    
- psf normalization
    - should each psf, before using it for deconvolution be scaled to sum to 1?
    - BIG psfgenerator sets max to 1




