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

class LabeledDeconvolutionExecutor : public StandardDeconvolutionExecutor {
public:
    LabeledDeconvolutionExecutor();
    virtual ~LabeledDeconvolutionExecutor() = default;





protected:
    // Helper methods for execution
    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& taskDesc,
        const ImageReader& reader,
        const ImageWriter& writer);


};