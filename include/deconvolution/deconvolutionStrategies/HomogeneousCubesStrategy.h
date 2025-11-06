#pragma once
#include "DeconvolutionStrategy.h"


// useful for simple deconvolution like apply these psfs to the entire image
// should be the best strtegy for speed, using entire memory and threads and optimizing cubes to have as little as possible -> less padding overhead
class HomogeneousCubesStrategy : public DeconvolutionStrategy{
public:

    ImageMap<std::vector<std::shared_ptr<PSF>>> getStrategy(
        const std::vector<PSF>& psfs,
        const RectangleShape imageShape,
        const int channelNumber,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm) override;

private:
    void validateConfiguration(
        const std::vector<PSF>& psfs,
        const RectangleShape imageShape,
        const int channelNumber,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm) const;
    
    std::vector<BoxCoord> splitImageHomogeneous(
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape);
    
    size_t getMemoryPerCube(
        size_t maxNumberThreads, 
        size_t maxMemory,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm);
    
    RectangleShape getCubeShape(
        size_t memoryPerCube,
        size_t numberThreads,
        RectangleShape imageOriginalShape,
        RectangleShape padding);

    ImageMap<std::vector<std::shared_ptr<PSF>>> addPSFS(
        std::vector<BoxCoord>& coords, 
        const std::vector<PSF>& psfs);
};