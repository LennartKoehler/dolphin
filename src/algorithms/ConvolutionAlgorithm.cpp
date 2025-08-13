#include "ConvolutionAlgorithm.h"


Hyperstack ConvolutionAlgorithm::run(Hyperstack& data, std::vector<PSF>& psfs){
    PSF psf; //TODO i think this is how it was in the main file before? doesnt make a lot of sense
    return data.convolve(psf);
}
