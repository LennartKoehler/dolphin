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

#include <string>
#include <vector>

// Canonical list of deconvolution algorithm names: the value stored in
// DeconvolutionConfig::algorithmName and the key used by
// DeconvolutionAlgorithmFactory.
//
// This lives outside the factory on purpose. DeconvolutionConfig registers
// algorithm_name as a StringSelection validated against this list, while the
// factory registers a config creator per name. If the list were taken from the
// factory, DeconvolutionConfig.cpp would have to include the factory, which in
// turn includes every concrete config -> include cycle. Keep this header free of
// any other dependency.
namespace DeconvolutionAlgorithms {

inline constexpr const char* kRichardsonLucy = "RichardsonLucy";
inline constexpr const char* kConvolution = "Convolution";
inline constexpr const char* kInverseFilter = "InverseFilter";
inline constexpr const char* kRichardsonLucyTotalVariation = "RichardsonLucyTotalVariation";
inline constexpr const char* kRegularizedInverseFilter = "RegularizedInverseFilter";
inline constexpr const char* kRichardsonLucyAdaptiveDamping = "RichardsonLucywithAdaptiveDamping";
inline constexpr const char* kPoissonMLEAdaptiveStep = "PoissonMLEAdaptiveStep";
inline constexpr const char* kPoissonMLE = "PoissonMLE";

inline const std::vector<std::string>& names() {
    static const std::vector<std::string> all = {
        kRichardsonLucy,
        kConvolution,
        kInverseFilter,
        kRichardsonLucyTotalVariation,
        kRegularizedInverseFilter,
        kRichardsonLucyAdaptiveDamping,
        kPoissonMLEAdaptiveStep,
        kPoissonMLE,
    };
    return all;
}

} // namespace DeconvolutionAlgorithms
