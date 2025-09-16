#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "backend/IDeconvolutionBackend.h"
#include "DeconvolutionAlgorithmFactory.h"

#include <fftw3.h>




struct CubeArrangement{
    int cubesPerX;      // Number of cubes along X axis
    int cubesPerY;      // Number of cubes along Y axis
    int cubesPerZ;      // Number of cubes along Z axis
    int cubesPerLayer;  // Number of cubes per layer (cubesPerX * cubesPerY)
    int totalGridNum;   // Total number of cubes (cubesPerX * cubesPerY * cubesPerZ)
    int cubePadding;

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

    void preprocess(const Hyperstack& input, const std::vector<PSF>& psfs);
    std::vector<cv::Mat> postprocessChannel(ImageMetaData& metaData, std::vector<std::vector<cv::Mat>>& gridImages);
    // Override base virtual methods to separate concerns
    virtual void configure(DeconvolutionConfig config);

protected:

    std::unique_ptr<IDeconvolutionBackend> backend_;
    std::unique_ptr<DeconvolutionAlgorithm> algorithm_;

    DeconvolutionConfig config;

    //memory
    fftw_plan forwardPlan  = nullptr;
    fftw_plan backwardPlan = nullptr;

    fftw_complex *fftwPlanMem = nullptr;

    //multiple psfs
    std::vector<fftw_complex*> preparedpsfs;
    RangeMap<fftw_complex*> layerPreparedPSFMap;
    RangeMap<fftw_complex*> cubePreparedPSFMap;

    //shapes
    RectangleShape cubeShape; // = psf padded shape
    RectangleShape psfOriginalShape;
    RectangleShape imageOriginalShape;
    CubeArrangement cubes;
    int cubePadding;

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
    std::vector<fftw_complex*> selectPSFsForCube(int cubeIndex);


private:
    void deconvolveSingleCube(int cubeIndex, std::vector<cv::Mat>& cubeImage);
    void deconvolveSingleCubePSF(fftw_complex* psf, std::vector<cv::Mat>& cubeImage);

    bool configured = false;
    // Internal helper functions
    void initPSFMaps(const std::vector<PSF>& psfs);
    void setupCubeArrangement();
    bool validateImageAndPsfSizes();
    void printConfigurationSummary() const;
    bool setupFFTWPlans();
    int getLayerIndex(int cubeIndex, int cubesPerLayer);
};