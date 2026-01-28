/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include "dolphin/deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphin/psf/PSF.h"
#include "dolphin/deconvolution/DeconvolutionAlgorithmFactory.h"
#include "dolphin/deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "dolphin/ThreadPool.h"
#include "dolphin/deconvolution/Preprocessor.h"
#include "dolphin/deconvolution/deconvolutionStrategies/DeconvolutionPlan.h"
#include "dolphin/deconvolution/DeconvolutionProcessor.h"
#include "dolphin/IO/TiffReader.h"
#include "dolphin/IO/TiffWriter.h"
#include "dolphin/backend/BackendFactory.h"
#include "dolphinbackend/IBackend.h"
#include "dolphinbackend/IBackendMemoryManager.h"

class SetupConfig;

/*
DeconvnolutionExecutor that takes a labelimage. This allows for different psfs for different parts of the image. 
The labelimage (int pixelvalues) provides the information for which psf should be used for each psf.
*/
class LabeledDeconvolutionExecutor : public StandardDeconvolutionExecutor {
public:
    LabeledDeconvolutionExecutor();
    virtual ~LabeledDeconvolutionExecutor() = default;

    // Configuration methods
    virtual void configure(const SetupConfig& setupConfig);
    virtual void configure(std::unique_ptr<DeconvolutionConfig> config);

    void setLabelReader(std::unique_ptr<ImageReader> labelReader) {this->labelReader = std::move(labelReader);}
    void setPsfLabelMap(RangeMap<std::string> psfLabelMap) {this->psfLabelMap = psfLabelMap;}


protected:
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc) override;

    std::vector<Label> getLabelGroups(
		const BoxCoord& roi,
		const std::vector<std::shared_ptr<PSF>>& psfs,
		const Image3D& image,
		RangeMap<std::string> psfLabelMap);

    std::vector<std::shared_ptr<PSF>> getPSFForLabel(Range<std::string>& psfids, const std::vector<std::shared_ptr<PSF>>& psfs);

    RangeMap<std::string> psfLabelMap;
    std::unique_ptr<ImageReader> labelReader;


    int featheringRadius = 0;
};