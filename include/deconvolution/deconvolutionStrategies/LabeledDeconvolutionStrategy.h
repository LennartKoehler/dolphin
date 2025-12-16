#pragma once
#include "StandardDeconvolutionStrategy.h"
#include "deconvolution/DeconvolutionConfig.h"
#include "../../HyperstackImage.h"
#include "../../psf/PSF.h"
#include "../DeconvolutionAlgorithmFactory.h"
#include "../algorithms/DeconvolutionAlgorithm.h"
#include "../../ThreadPool.h"
#include "../Preprocessor.h"
#include "ComputationalPlan.h"
#include "../DeconvolutionProcessor.h"
#include "../../IO/TiffReader.h"
#include "../../IO/TiffWriter.h"
#include "../../backend/BackendFactory.h"
#include "../../backend/IBackend.h"
#include "../../backend/IBackendMemoryManager.h"

// Forward declaration
class SetupConfig;

class LabeledDeconvolutionStrategy : public StandardDeconvolutionStrategy {
public:
    LabeledDeconvolutionStrategy() = default;
    virtual ~LabeledDeconvolutionStrategy() = default;

    // IDeconvolutionStrategy interface
    virtual void configure(const SetupConfig& setupConfig) override;
    
    virtual ChannelPlan createPlan(
        const ImageMetaData& metadata,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config) override;


    // ILabeledDeconvolutionStrategy interface
    virtual void setLabeledImage(std::shared_ptr<Hyperstack> labelImage);
    virtual void setLabelPSFMap(const RangeMap<std::string>& labelPSFMap);

protected:
    // Helper methods for plan creation
    virtual std::vector<Label> getLabelGroups(int channelNumber, const BoxCoord& roi, std::vector<std::shared_ptr<PSF>>& psfs);
    virtual std::vector<std::shared_ptr<PSF>> getPSFForLabel(Range<std::string>& psfids, std::vector<std::shared_ptr<PSF>>& psfs);

    // Reuse methods from StandardDeconvolutionStrategy
    // virtual size_t maxMemoryPerCube(
    //     size_t maxNumberThreads, 
    //     size_t maxMemory,
    //     const std::shared_ptr<DeconvolutionAlgorithm> algorithm);

    // virtual size_t estimateMemoryUsage(
    //     const RectangleShape& cubeSize,
    //     const std::shared_ptr<DeconvolutionAlgorithm> algorithm);
    
    // virtual RectangleShape getCubeShape(
    //     size_t memoryPerCube,
    //     size_t numberThreads,
    //     const RectangleShape& imageOriginalShape,
    //     const Padding& cubePadding);

    virtual Padding getImagePadding(
        const RectangleShape& imageSize,
        const RectangleShape& cubeSizeUnpadded,
        const Padding& cubePadding
    );

    virtual Padding getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs);

private:
    RangeMap<std::string> psfLabelMap;
    std::shared_ptr<Hyperstack> labelImage;

};