#pragma once
#include "StandardDeconvolutionStrategy.h"


class LabeledImageDeconvolutionStrategy : public StandardDeconvolutionStrategy{

public:
    // Override setter methods for labeled deconvolution
    void setLabeledImage(std::shared_ptr<Hyperstack> labelImage);
    void setLabelPSFMap(const RangeMap<std::string>& labelPSFMap);

private:
    ComputationalPlan createComputationalPlan(
        int channelNumber,
        const Image3D& image,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm) override ;

    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& task,
        const PaddedImage& inputImagePadded,
        std::vector<cv::Mat>& outputImage,
        std::mutex& writerMutex) override ;
    
    virtual std::vector<LabelGroup> getLabelGroups(int channelumber, const BoxCoord& roi, std::vector<std::shared_ptr<PSF>>& psfs);
    std::vector<std::shared_ptr<PSF>> getPSFForLabel(int label, std::vector<std::shared_ptr<PSF>>& psfs);

std::shared_ptr<Hyperstack> labelImage;
RangeMap<std::string> psfLabelMap;
std::vector<std::shared_ptr<PSF>> psfs;
};