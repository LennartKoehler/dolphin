#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/InverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RegularizedInverseFilterDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLTVDeconvolutionAlgorithm.h"
#include "deconvolution/algorithms/RLADDeconvolutionAlgorithm.h"

#ifdef CUDA_AVAILABLE
#include "deconvolution/algorithms/BaseDeconvolutionAlgorithmGPU.h"
#endif

DeconvolutionAlgorithmFactory& DeconvolutionAlgorithmFactory::getInstance() {
    static DeconvolutionAlgorithmFactory instance;
    return instance;
}

DeconvolutionAlgorithmFactory::DeconvolutionAlgorithmFactory() {
    // Detect CUDA availability first
    // Register all CPU algorithms
    registerCPUAlgorithms();
    
    // Register GPU algorithms if CUDA is available
    if (is_cuda_available_) {
        registerGPUAlgorithms();
        std::cout << "[INFO] CUDA available, GPU algorithms registered" << std::endl;
    } else {
        std::cout << "[INFO] CUDA not available, only CPU algorithms registered" << std::endl;
    }
}