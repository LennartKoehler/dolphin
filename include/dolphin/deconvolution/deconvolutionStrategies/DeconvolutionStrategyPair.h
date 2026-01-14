#pragma once

#include "IDeconvolutionStrategy.h"
#include "IDeconvolutionExecutor.h"
#include <memory>

/**
 * A simple container that holds both a deconvolution strategy and its corresponding executor.
 * This class contains no additional functionality beyond holding the strategy and executor.
 */
class DeconvolutionStrategyPair {
public:
    /**
     * Create a strategy pair with strategy and executor
     * @param strategy The deconvolution strategy
     * @param executor The corresponding executor
     */
    DeconvolutionStrategyPair(
        std::unique_ptr<IDeconvolutionStrategy> strategy,
        std::unique_ptr<IDeconvolutionExecutor> executor);
    
    ~DeconvolutionStrategyPair() = default;
    
    // Delete copy constructor and assignment operator
    DeconvolutionStrategyPair(const DeconvolutionStrategyPair&) = delete;
    DeconvolutionStrategyPair& operator=(const DeconvolutionStrategyPair&) = delete;
    
    // Move constructor and assignment operator
    DeconvolutionStrategyPair(DeconvolutionStrategyPair&&) = default;
    DeconvolutionStrategyPair& operator=(DeconvolutionStrategyPair&&) = default;
    
    /**
     * Get the strategy (non-owning reference)
     * @return Reference to the strategy
     */
    IDeconvolutionStrategy& getStrategy() { return *strategy_; }
    
    /**
     * Get the executor (non-owning reference)
     * @return Reference to the executor
     */
    IDeconvolutionExecutor& getExecutor() { return *executor_; }
    
    /**
     * Get the strategy (const non-owning reference)
     * @return Const reference to the strategy
     */
    const IDeconvolutionStrategy& getStrategy() const { return *strategy_; }
    
    /**
     * Get the executor (const non-owning reference)
     * @return Const reference to the executor
     */
    const IDeconvolutionExecutor& getExecutor() const { return *executor_; }

private:
    std::unique_ptr<IDeconvolutionStrategy> strategy_;
    std::unique_ptr<IDeconvolutionExecutor> executor_;
};