#include "frontend/gui/UIDeconvolutionConfig.h"
#include "frontend/gui/imguiWidget.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
UIDeconvolutionConfig::UIDeconvolutionConfig(){
    setParameters(std::make_shared<DeconvolutionConfig>());
}

void UIDeconvolutionConfig::setParameters(std::shared_ptr<const DeconvolutionConfig> config){
    auto configCopy = std::make_shared<DeconvolutionConfig>(*config);
    deconvolutionConfig = configCopy;
    setDeconvolutionConfigParameters(configCopy);
}

std::shared_ptr<DeconvolutionConfig> UIDeconvolutionConfig::getConfig(){
    return deconvolutionConfig;
}


void UIDeconvolutionConfig::setDeconvolutionConfigParameters(std::shared_ptr<DeconvolutionConfig> deconvolutionConfig){
    static std::vector<std::string> algorithmOptions = DeconvolutionAlgorithmFactory::getInstance().getAvailableAlgorithms();
    static StringSelectionHelper algorithmHelper{&deconvolutionConfig->algorithmName, &algorithmOptions};
    std::vector<ParameterDescription> runtimeParams = {
        // Core algorithm parameters

        {"Algorithm", ParameterType::VectorString, &algorithmHelper, 0.0, 0.0},
        {"Iterations", ParameterType::Int, &deconvolutionConfig->iterations, 1, 10000},
        {"Lambda", ParameterType::Double, &deconvolutionConfig->lambda, 0.0, 1.0},
        {"Epsilon", ParameterType::Double, &deconvolutionConfig->epsilon, 1e-12, 1e-3},
        
        // Processing parameters
        {"Grid Processing", ParameterType::Bool, &deconvolutionConfig->grid, 0.0, 1.0},
        {"Subimage Size", ParameterType::Int, &deconvolutionConfig->subimageSize, 0, 10000},
        {"Border Type", ParameterType::Int, &deconvolutionConfig->borderType, 0, 5},
        


        // not supported in gui yet
        // {"Second PSF Layers", ParameterType::Map, &deconvolutionConfig->layerPSFMap, 0.0, 0.0},
        // {"Second PSF Cubes", ParameterType::Map, &deconvolutionConfig->cubePSFMap, 0.0, 0.0},
    };
    parameters.insert(parameters.end(), runtimeParams.begin(), runtimeParams.end());
}