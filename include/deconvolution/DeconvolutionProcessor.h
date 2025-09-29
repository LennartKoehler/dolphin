#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "IDeconvolutionBackend.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "deconvolution/algorithms/DeconvolutionAlgorithm.h"



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
    std::shared_ptr<IDeconvolutionBackend> backend_; // should this be shared?
    std::shared_ptr<DeconvolutionAlgorithm> algorithm_;

    DeconvolutionConfig config;



    //multiple psfs
    std::vector<complex*> preparedpsfs;
    RangeMap<complex*> layerPreparedPSFMap;
    RangeMap<complex*> cubePreparedPSFMap;

    //shapes
    RectangleShape subimageShape; // before padding but after splitting
    RectangleShape psfOriginalShape;
    RectangleShape imageOriginalShape;
    RectangleShape cubeShapePadded; // dims both subimages/images and psf are during computation
    CubeArrangement cubes;

 

    void preprocess(const Hyperstack& input, const std::vector<PSF>& psfs);
    std::vector<cv::Mat> postprocessChannel(ImageMetaData& metaData, std::vector<std::vector<cv::Mat>>& gridImages);
    
    // Helper functions that don't depend on execution backend

    /**
     * @brief Prepare PSF for processing - performs FFT and padding
     * @param psfs List of PSFs to prepare
     * @return true if PSF preparation succeeded, false otherwise
     */
    void preprocessPSF(const std::vector<PSF>& inputPSFs);
    std::vector<std::vector<cv::Mat>> preprocessChannel(Channel& channel);

    void setPSFShape(const PSF& psf);
    void setImageOriginalShape(const Channel& channel);
    void setCubeShape(
        const RectangleShape& imageOriginalShape,
        bool configgrid,
        int configcubeSize,
        int configpsfSafetyBorder
    );

    /**
     * @brief Select appropriate PSF for current grid image based on layer and cube mappings
     * @param gridImageIndex Index of current grid image
     * @return Pointer to selected PSF's padded FFTW complex array
     */
    std::vector<complex*> selectPSFsForCube(int cubeIndex);


private:
    void deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage);
    void deconvolveSingleCubePSF(complex* psf, std::vector<cv::Mat>& cubeImage);
    void initPSFMaps(const std::vector<PSF>& psfs);
    std::shared_ptr<IDeconvolutionBackend> loadBackend(const std::string& backendName);
    void setupCubeArrangement();
    bool validateImageAndPsfSizes();
    int getLayerIndex(int cubeIndex, int cubesPerLayer);
    void padPSF(const ComplexData& psf, ComplexData& padded_psf);


    bool configured = false;

};



// class DeconvolutionProcessorParallel : public DeconvolutionProcessor{
//     virtual void deconvolveSingleCubePSF(complex* psf, std::vector<cv::Mat>& cubeImage) override;
//     std::shared_ptr<IDeconvolutionBackend> getThreadLocalBackend();

//     thread_local static std::shared_ptr<IDeconvolutionBackend> thread_backend_;

// };