#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "IDeconvolutionBackend.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "deconvolution/ThreadManager.h"


struct CubeArrangement{
    int cubesPerX;      // Number of cubes along X axis
    int cubesPerY;      // Number of cubes along Y axis
    int cubesPerZ;      // Number of cubes along Z axis
    int cubesPerLayer;  // Number of cubes per layer (cubesPerX * cubesPerY)
    int totalGridNum;   // Total number of cubes (cubesPerX * cubesPerY * cubesPerZ)

};

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
    Hyperstack run(Hyperstack& input, const std::vector<PSF>& psfs); // careful, this edits input inplace

    virtual ~DeconvolutionProcessor() { cleanup(); }
    void cleanup();


    virtual void configure(DeconvolutionConfig config);

protected:
    std::shared_ptr<IDeconvolutionBackend> cpu_backend_;
    std::shared_ptr<IDeconvolutionBackend> backend_; //TODO never initialized
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;

    DeconvolutionConfig config;



    //multiple psfs
    std::vector<ComplexData> preparedpsfs;
    RangeMap<ComplexData> layerPreparedPSFMap;
    RangeMap<ComplexData> cubePreparedPSFMap;

    //shapes
    RectangleShape subimageShape; // before padding but after splitting
    RectangleShape psfOriginalShape;
    RectangleShape imageOriginalShape;
    RectangleShape imageShapePadded;
    RectangleShape cubeShapePadded; // dims both subimages/images and psf are during computation
    CubeArrangement cubes;

 

    void init(const Hyperstack& input, const std::vector<PSF>& psfs);
    std::vector<cv::Mat> postprocessChannel(ImageMetaData& metaData, const std::vector<std::vector<cv::Mat>>& gridImages);
    
    void preprocessPSF(std::vector<PSF> inputPSFs);
    std::vector<std::vector<cv::Mat>> preprocessChannel(Channel& channel);

    void setPSFOriginalShape(const PSF& psf);
    void setImageOriginalShape(const Channel& channel);
    void setCubeShape(
        const RectangleShape& imageOriginalShape,
        bool configgrid,
        const RectangleShape& subimageSize
    );
    void addPaddingToShapes(const RectangleShape& padding);

    const std::vector<ComplexData> selectPSFsForCube(int cubeIndex);


private:
    std::future<ComplexData> deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage);
    void initPSFMaps(const std::vector<PSF>& psfs);
    std::shared_ptr<IDeconvolutionBackend> loadBackend(const std::string& backendName);
    void setupCubeArrangement();
    int getLayerIndex(int cubeIndex, int cubesPerLayer);
    void parallelDeconvolution(std::vector<std::vector<cv::Mat>>& cubeImages);

    std::unique_ptr<ThreadManager> threadManager_;
    bool configured = false;

};



// class DeconvolutionProcessorParallel : public DeconvolutionProcessor{
//     virtual void deconvolveSingleCubePSF(complex* psf, std::vector<cv::Mat>& cubeImage) override;
//     std::shared_ptr<IDeconvolutionBackend> getThreadLocalBackend();

//     thread_local static std::shared_ptr<IDeconvolutionBackend> thread_backend_;

// };