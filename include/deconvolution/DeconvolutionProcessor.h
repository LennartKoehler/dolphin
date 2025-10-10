#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"

struct CubeArrangement{
    int cubesPerX;      // Number of cubes along X axis
    int cubesPerY;      // Number of cubes along Y axis
    int cubesPerZ;      // Number of cubes along Z axis
    int cubesPerLayer;  // Number of cubes per layer (cubesPerX * cubesPerY)
    int totalGridNum;   // Total number of cubes (cubesPerX * cubesPerY * cubesPerZ)

};

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
    Hyperstack run(Hyperstack& input, const std::vector<PSF>& psfs); // careful, this edits input inplace

    virtual ~DeconvolutionProcessor() { cleanup(); }
    void cleanup();


    virtual void configure(const DeconvolutionConfig config);


private:
 
    void init(const Hyperstack& input, const std::vector<PSF>& psfs);
    void parallelDeconvolution(std::vector<std::vector<cv::Mat>>& cubeImages);
    void deconvolveSingleCube(
        std::shared_ptr<IBackend> backend,
        std::unique_ptr<DeconvolutionAlgorithm> algorithm,
        std::vector<cv::Mat>& cubeImage,
        const RectangleShape& workShape,
        const std::vector<ComplexData>& psfs_host);

    std::vector<std::vector<cv::Mat>> preprocessChannel(Channel& channel);

    std::vector<cv::Mat> postprocessChannel(ImageMetaData& metaData, const std::vector<std::vector<cv::Mat>>& gridImages);

    void setPSFOriginalShape(const PSF& psf);
    void setImageOriginalShape(const Channel& channel);
    void setWorkShapes(
        const RectangleShape& imageOriginalShape,
        const RectangleShape& padding,
        size_t subimageSize);
    size_t getMemoryPerCube(size_t maxNumberThreads); 
    void setupCubeArrangement();

    void preprocessPSF(std::vector<PSF> inputPSFs);
    const std::vector<ComplexData> selectPSFsForCube(int cubeIndex);
    void initPSFMaps(const std::vector<PSF>& psfs);
    int getLayerIndex(int cubeIndex, int cubesPerLayer);


    std::shared_ptr<IBackend> loadBackend(const std::string& backendName);
 
    ComplexData convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, const RectangleShape& shape);
    std::vector<cv::Mat> convertFFTWComplexToCVMatVector(const ComplexData& input);



    DeconvolutionConfig config;

    std::shared_ptr<IBackend> cpu_backend_;
    std::shared_ptr<IBackend> backend_;
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;


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

    //multithreading
    std::shared_ptr<ThreadPool> threadPool;
    size_t numberThreads;

    bool configured = false;

};



