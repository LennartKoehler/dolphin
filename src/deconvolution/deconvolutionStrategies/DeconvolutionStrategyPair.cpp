#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategyPair.h"

DeconvolutionStrategyPair::DeconvolutionStrategyPair(
    std::unique_ptr<IDeconvolutionStrategy> strategy,
    std::unique_ptr<IDeconvolutionExecutor> executor)
    : strategy_(std::move(strategy)), executor_(std::move(executor)) {
    // Constructor body is empty - just move the unique pointers
}