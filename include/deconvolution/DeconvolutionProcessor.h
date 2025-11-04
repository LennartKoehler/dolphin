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
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"
#include "ImageMap.h"
#include "Preprocessor.h"



class IBackend;

/**
 * Abstract base class for deconvolution algorithms that separates common,
 * execution-agnostic functionality from backend-specific operations.
 * 
 * - Common grid processing logic
 * - PSF mapping and selection
 * - Data structure management
 * - Main orchestration methods
 * - Platform-independent helper functions
 * 
 * Backend-specific operations are left as pure virtual methods to be
 * implemented by concrete algorithm classes (CPU, GPU, etc.).
 */
class DeconvolutionProcessor{
public:
    Hyperstack run(Hyperstack& input, ImageMap<std::shared_ptr<PSF>>& psfMap); // careful, this edits input inplace

    DeconvolutionProcessor();


    virtual void configure(const DeconvolutionConfig config);


private:
 
    void init(const Hyperstack& input, ImageMap<std::shared_ptr<PSF>>& psfs);
    void parallelDeconvolution(std::vector<cv::Mat>& image, ImageMap<std::shared_ptr<PSF>>& psfMap);
    void deconvolveSingleCube(
        std::shared_ptr<IBackend> backend,
        std::unique_ptr<DeconvolutionAlgorithm> algorithm,
        std::vector<cv::Mat>& cubeImage,
        const RectangleShape& workShape,
        const std::vector<const ComplexData*> psfs_host);


    void postprocessChannel(ImageMetaData& metaData, std::vector<cv::Mat>& image);
    RectangleShape getCubePadding(BoxCoord box);

    void setImageOriginalShape(const Channel& channel);
    void setImageShapePadded(const ImageMap<std::shared_ptr<PSF>>& psfs);
    RectangleShape getPadding(const ImageMap<std::shared_ptr<PSF>>& psfs);



    std::vector<cv::Mat> getCubeImage(const std::vector<cv::Mat>& image, BoxCoord box, RectangleShape workShape);

 
    ComplexData convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, const RectangleShape& shape);
    std::vector<cv::Mat> convertFFTWComplexToCVMatVector(const ComplexData& input);



    DeconvolutionConfig config;

    std::shared_ptr<IBackendMemoryManager> cpuMemoryManager;
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;

    PSFPreprocessor psfPreprocessor;




    //shapes
    RectangleShape imageOriginalShape;
    RectangleShape imageShapePadded;


    //multithreading
    std::shared_ptr<ThreadPool> threadPool;
    size_t numberThreads;

    bool configured = false;

};



