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

#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategyPair.h"

DeconvolutionStrategyPair::DeconvolutionStrategyPair(
    std::unique_ptr<IDeconvolutionStrategy> strategy,
    std::unique_ptr<IDeconvolutionExecutor> executor)
    : strategy_(std::move(strategy)), executor_(std::move(executor)) {
    // Constructor body is empty - just move the unique pointers
}