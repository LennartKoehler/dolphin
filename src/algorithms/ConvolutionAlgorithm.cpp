#include "ConvolutionAlgorithm.h"


Hyperstack ConvolutionAlgorithm::run(Hyperstack& data, std::vector<PSF>& psfs){
    return data.convolve(psfs[0]);
}
