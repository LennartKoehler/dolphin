#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategyFacade.h"
#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/StandardDeconvolutionExecutor.h"
#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/LabeledDeconvolutionExecutor.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "backend/BackendFactory.h"
#include "backend/IBackend.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include <stdexcept>

DeconvolutionStrategyFacade::DeconvolutionStrategyFacade() {
    // Create strategy and executor instances
    standardStrategy = std::make_unique<StandardDeconvolutionStrategy>();
    standardExecutor = std::make_unique<StandardDeconvolutionExecutor>();
    labeledStrategy = std::make_unique<LabeledDeconvolutionStrategy>();
    labeledExecutor = std::make_unique<LabeledDeconvolutionExecutor>();
}

Hyperstack DeconvolutionStrategyFacade::run(const ImageReader& reader, const std::vector<PSF>& inputPSFS) {
    if (!configured) {
        throw std::runtime_error("Strategy must be configured before running");
    }

    // Create computational plan using the strategy
    ChannelPlan plan;
    ImageMetaData metadata; // This would come from the reader in a real implementation
    metadata.imageWidth = 512; // Placeholder values
    metadata.imageLength = 512;
    metadata.slices = 10;
    
    DeconvolutionAlgorithmFactory& fact = DeconvolutionAlgorithmFactory::getInstance();
    std::shared_ptr<DeconvolutionAlgorithm> algorithm = fact.createShared(*standardStrategy->config);

    BackendFactory& bf = BackendFactory::getInstance();
    std::shared_ptr<IBackend> backend = bf.createShared(standardStrategy->config->backenddeconv);

    if (useLabeled) {
        plan = labeledStrategy->createPlan(metadata, inputPSFS, *standardStrategy->config, backend, algorithm);
    } else {
        plan = standardStrategy->createPlan(metadata, inputPSFS, *standardStrategy->config, backend, algorithm);
    }

    // Execute the plan using the executor
    if (useLabeled) {
        return labeledExecutor->execute(plan);
    } else {
        return standardExecutor->execute(plan);
    }
}

void DeconvolutionStrategyFacade::configure(std::unique_ptr<DeconvolutionConfig> config) {
    // Configure both strategy and executor
    standardStrategy->configure(std::move(config));
    standardExecutor->configure(standardStrategy->config->clone());
    
    if (labeledStrategy && labeledExecutor) {
        labeledStrategy->configure(standardStrategy->config->clone());
        labeledExecutor->configure(standardStrategy->config->clone());
    }
    
    configured = true;
}

// Factory method implementation
std::unique_ptr<DeconvolutionStrategyFacade> DeconvolutionStrategyFacade::create(bool useLabeled) {
    auto facade = std::make_unique<DeconvolutionStrategyFacade>();
    facade->useLabeled = useLabeled;
    return facade;
}