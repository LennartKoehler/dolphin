#pragma once
#include "deconvolution/ImageMap.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "HyperstackImage.h"

class DeconvolutionAlgorithm;
class IBackend;


// Creates the imagemap, which maps cubes of the image to a vector of speicific psfs that should be used for deconvolution of this specific cube
// takes in all information that oculd possibly be useful for creating such a strategy, but should not actually store any data like psfs or images
// but is rather a lightweight object that only defines how it should be processed, not really what

// strategies might include one that simply is optimized for performance, e.g. deconvolve the entire image with this one psf
//      and let the strategy decide how to create cubes to make the deconvolutionprocessor be most efficient
// another example would be that the strategy runs a gui and lets the user define the cubes and the mapping of cube to psf. so the user decides the cubes, and some other optimization
//      parts are hardcoded into the strategy
class DeconvolutionStrategy{
public:
    DeconvolutionStrategy() = default;
    virtual ~DeconvolutionStrategy(){}

    virtual ImageMap<std::vector<std::shared_ptr<PSF>>> getStrategy(
        const std::vector<PSF>& psfs,
        const RectangleShape imageShape,
        const int channelNumber,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm) = 0;

};