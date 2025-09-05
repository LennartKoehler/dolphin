// #include "deconvolution/DeconvolutionAlgorithmFactory.h"
// #include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
// #include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"

// DeconvolutionAlgorithmFactory& DeconvolutionAlgorithmFactory::getInstance() {
//     static DeconvolutionAlgorithmFactory instance;
//     return instance;
// }

// DeconvolutionAlgorithmFactory::DeconvolutionAlgorithmFactory() {
//     // Register algorithms
//     registerAlgorithm("InverseFilter", []() {
//         return std::make_unique<InverseFilterDeconvolutionAlgorithm>();
//     });
//     registerAlgorithm("RichardsonLucy", []() {
//         return std::make_unique<RLDeconvolutionAlgorithm>();
//     });
//     registerAlgorithm("RichardsonLucyTotalVariation", []() {
//         return std::make_unique<RLTVDeconvolutionAlgorithm>();
//     });
//     registerAlgorithm("RegularizedInverseFilter", []() {
//         return std::make_unique<RegularizedInverseFilterDeconvolutionAlgorithm>();
//     });
//     registerAlgorithm("RichardsonLucywithAdaptiveDamping", []() {
//         return std::make_unique<RLADDeconvolutionAlgorithm>();
//     });
// }