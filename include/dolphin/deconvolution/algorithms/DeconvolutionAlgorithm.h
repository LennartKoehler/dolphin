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
#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphinbackend/IBackend.h"
#include <functional>
#include <stdexcept>
#include <string>

class DeconvolutionAlgorithm{
    using prFunction = std::function<void(int)>;
public:
    DeconvolutionAlgorithm() = default;
    virtual ~DeconvolutionAlgorithm() = default;

    /**
     * @brief Configure the algorithm from its own config class.
     *
     * The factory entry point (see DeconvolutionAlgorithmFactory::create) hands
     * the base reference to configure(const DeconvolutionConfig&), which
     * downcasts to the algorithm's own config class and forwards here. Each
     * algorithm therefore reads exactly the parameters its config registers.
     */
    virtual void configure(const DeconvolutionConfig& config) = 0;


    // Initialize algorithm-specific data allocations
    virtual void init(const CuboidShape& dataSize) = 0;
    virtual bool isInitialized() const = 0;

    // it is assumed that the input of convolve is already located on the backend device
    virtual void deconvolve(const ComplexData& H, RealData& g, RealData& f) = 0;

    void setBackend(IBackend& backend){this->backend = &backend;}

    inline std::unique_ptr<DeconvolutionAlgorithm> clone() const{
        std::unique_ptr<DeconvolutionAlgorithm> clone = cloneSpecific();
        if (progressFunction)
            clone->setProgressTracker(progressFunction);
        return clone;
    }

    virtual size_t getMemoryMultiplier() const = 0;
    void setProgressTracker(prFunction progressFunction){ this->progressFunction = progressFunction; }
protected:
    virtual std::unique_ptr<DeconvolutionAlgorithm> cloneSpecific() const = 0;

    /**
     * @brief Downcast a base config to the algorithm's config class.
     * @throws std::runtime_error naming both the algorithm and the config class
     */
    template <typename DerivedConfig>
    static const DerivedConfig& asConfig(const DeconvolutionConfig& config, const char* algorithmName) {
        const auto* typed = dynamic_cast<const DerivedConfig*>(&config);
        if (!typed) {
            throw std::runtime_error(std::string("Algorithm '") + algorithmName +
                                     "' requires a " + DerivedConfig::kAlgorithmName +
                                     " config, got: " + config.getName());
        }
        return *typed;
    }

    // Shared numerical guard for division-based updates (set from the config's
    // min_denominator by the algorithms that use it).
    float min_denominator = 1e-6f;

    prFunction progressFunction;
    IBackend* backend;


};
