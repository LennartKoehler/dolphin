#pragma once
#include "DeconvolutionStrategy.h"
#include <string>
#include "deconvolution/DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "deconvolution/DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "deconvolution/Preprocessor.h"
#include "deconvolution/deconvolutionStrategies/DeconvolutionStrategy.h"
#include "deconvolution/deconvolutionStrategies/ComputationalPlan.h"
#include "deconvolution/DeconvolutionProcessor.h"



// useful for simple deconvolution like apply these psfs to the entire image
// should be the best strtegy for speed, using entire memory and threads and optimizing cubes to have as little as possible -> less padding overhead
class StandardDeconvolutionStrategy : public DeconvolutionStrategy{
public:




    // it is up to the client to make the psfMapp efficient, e.g. by using the appropriate cubeSize, and using as little different cube sizes as possible
    // the deconvolutionprocessor simply processes the input
    virtual Hyperstack run(const Hyperstack& image, const std::vector<PSF>& psfs) override;

    StandardDeconvolutionStrategy();

    virtual void configure(std::unique_ptr<DeconvolutionConfig> config) override;


protected:
 
    virtual void parallelDeconvolution(const PaddedImage& image, std::vector<cv::Mat>& output, const ComputationalPlan& channelPlan);

    virtual std::function<void()> createTask(
        const std::unique_ptr<CubeTaskDescriptor>& task,
        const PaddedImage& inputImagePadded,
        std::vector<cv::Mat>& outputImage,
        std::mutex& writerMutex);



    virtual PaddedImage preprocessChannel(Channel& image, const ComputationalPlan& channelPlan);
    virtual void postprocessChannel(Channel& image);
    virtual Padding getCubePadding(const RectangleShape& image, const std::vector<PSF> psfs);

    virtual PaddedImage getCubeImage(const PaddedImage& image, const BoxCoord& srcbox, const Padding& cubePadding);

        
    virtual size_t memoryForShape(const RectangleShape& shape);

    virtual ComplexData convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, const RectangleShape& shape);
    virtual std::vector<cv::Mat> convertFFTWComplexToCVMatVector(const ComplexData& input);

    //--------------------------------------------------------------------------------------------------
    virtual ComputationalPlan createComputationalPlan(
        int channelNumber,
        const Channel& image,
        const std::vector<PSF>& psfs,
        const DeconvolutionConfig& config,
        const std::shared_ptr<IBackend> backend,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm); 


    
    size_t maxMemoryPerCube(
        size_t maxNumberThreads, 
        size_t maxMemory,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm);

    size_t estimateMemoryUsage(
        const RectangleShape& cubeSize,
        const std::shared_ptr<DeconvolutionAlgorithm> algorithm
    );
    
    RectangleShape getCubeShape(
        size_t memoryPerCube,
        size_t numberThreads,
        const RectangleShape& imageOriginalShape,
        const Padding& cubePadding);

    Padding getImagePadding(
        const RectangleShape& imageSize,
        const RectangleShape& cubeSize,
        const Padding& cubePadding
    );



protected:
    std::unique_ptr<DeconvolutionConfig> config;

    std::shared_ptr<IBackendMemoryManager> cpuMemoryManager;
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;

    PSFPreprocessor psfPreprocessor;

    LoadingBar loadingBar;

    PaddingStrategy paddingStrat;
    DeconvolutionProcessor processor;

    //multithreading
    std::shared_ptr<ThreadPool> readwriterPool;
    size_t numberThreads;

    bool configured = false;
    
};

