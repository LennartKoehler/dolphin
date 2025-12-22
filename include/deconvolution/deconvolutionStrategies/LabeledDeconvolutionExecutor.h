#pragma once
#include "StandardDeconvolutionExecutor.h"
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

class SetupConfig;

class LabeledDeconvolutionExecutor : public StandardDeconvolutionExecutor {
public:
    LabeledDeconvolutionExecutor();
    virtual ~LabeledDeconvolutionExecutor() = default;

    // Configuration methods
    virtual void configure(const SetupConfig& setupConfig);

    void setLabelReader(std::unique_ptr<ImageReader> labelReader) {this->labelReader = std::move(labelReader);}
    void setPsfLabelMap(RangeMap<std::string> psfLabelMap) {this->psfLabelMap = psfLabelMap;}


protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
        const ImageReader& reader,
        const ImageWriter& writer);

    std::vector<Label> getLabelGroups(
        int channelNumber,
		const BoxCoord& roi,
		const std::vector<std::shared_ptr<PSF>>& psfs,
		const Image3D& image,
		RangeMap<std::string> psfLabelMap);

    std::vector<std::shared_ptr<PSF>> getPSFForLabel(Range<std::string>& psfids, const std::vector<std::shared_ptr<PSF>>& psfs);

    RangeMap<std::string> psfLabelMap;
    std::unique_ptr<ImageReader> labelReader;

};