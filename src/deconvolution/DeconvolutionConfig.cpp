#include "deconvolution/DeconvolutionConfig.h"

DeconvolutionConfig::DeconvolutionConfig() {
    registerAllParameters();
}

void DeconvolutionConfig::registerAllParameters(){
    bool optional = true;
    
    // Register all deconvolution parameters
    registerParameter("algorithmName", algorithmName, !optional);  // Required
    registerParameter("subimageSize", subimageSize, optional);
    registerParameter("iterations", iterations, optional);
    registerParameter("epsilon", epsilon, optional);
    registerParameter("grid", grid, optional);
    registerParameter("lambda", lambda, optional);
    registerParameter("borderType", borderType, optional);
    registerParameter("psfSafetyBorder", psfSafetyBorder, optional);
    registerParameter("cubeSize", cubeSize, optional);
    registerParameter("secondpsflayers", secondpsflayers, optional);
    registerParameter("secondpsfcubes", secondpsfcubes, optional);
    
    // Commented out parameters - add if needed
    // registerParameter("time", time, optional);
    // registerParameter("saveSubimages", saveSubimages, optional);
    // registerParameter("gpu", gpu, optional);
}
DeconvolutionConfig::DeconvolutionConfig(const DeconvolutionConfig& other)
    : Config(other){
    registerAllParameters();
    algorithmName = other.algorithmName;
    subimageSize = other.subimageSize;
    iterations = other.iterations;
    epsilon = other.epsilon;
    grid = other.grid;
    lambda = other.lambda;
    borderType = other.borderType;
    psfSafetyBorder = other.psfSafetyBorder;
    cubeSize = other.cubeSize;
    secondpsflayers = other.secondpsflayers;
    secondpsfcubes = other.secondpsfcubes;
}