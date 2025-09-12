#pragma once
#include <string>
#include "DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "backend/IDeconvolutionBackend.h"
#include "algorithms/DeconvolutionAlgorithm.h"

#include <fftw3.h>
typedef fftw_complex PSFfftw;



struct RectangleShape{
    int width;
    int height;
    int depth;
    int volume;
};

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
 * This class inherits from BaseDeconvolutionAlgorithm and provides:
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
    
    virtual ~DeconvolutionProcessor() { cleanup(); }
    void cleanup();
    Hyperstack run(Hyperstack& input, const std::unordered_map<size_t, std::shared_ptr<PSF>>& psfs); // careful, this edits input inplace
    void preprocess(const Hyperstack& input, const std::unordered_map<size_t, std::shared_ptr<PSF>>& psfs);
    std::vector<cv::Mat> postprocessChannel(ImageMetaData& metaData, std::vector<std::vector<cv::Mat>>& gridImages);
    // Override base virtual methods to separate concerns
    virtual void configure(DeconvolutionConfig config);

    /**
     * @brief Backend-specific preprocessing operations
     * @param channel_num Channel number being processed
     * @param psf_index Index of PSF for current processing
     * @return true if preprocessing succeeded, false otherwise
     */
    virtual bool preprocessBackendSpecific(int channel_num, int psf_index) = 0;

    /**
     * @brief Backend-specific operations for main algorithm
     * @param channel_num Channel number being processed
     * @param H FFTW complex array for PSF (frequency domain)
     * @param g FFTW complex array for observed image (frequency domain)
     * @param f FFTW complex array for estimated image (frequency domain)
     */
    virtual void algorithmBackendSpecific(int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) = 0;

    /**
     * @brief Backend-specific postprocessing operations
     * @param channel_num Channel number being processed
     * @param psf_index Index of PSF for current processing
     * @return true if postprocessing succeeded, false otherwise
     */
    virtual bool postprocessBackendSpecific(int channel_num, int psf_index) = 0;

    /**
     * @brief Backend-specific memory allocation
     * @param channel_num Channel number being processed
     * @return true if memory allocation succeeded, false otherwise
     */
    virtual bool allocateBackendMemory(int channel_num) = 0;

    /**
     * @brief Backend-specific memory deallocation
     * @param channel_num Channel number being processed
     */
    virtual void deallocateBackendMemory(int channel_num) = 0;

    /**
     * @brief Backend-specific cleanup operations
     */
    virtual void cleanupBackendSpecific() = 0;
    
    /**
     * @brief Algorithm-specific configuration
     * @param config Deconvolution configuration
     */
    virtual void configureAlgorithmSpecific(const DeconvolutionConfig& config) = 0;

protected:

    std::unique_ptr<IDeconvolutionBackend> backend_;
    std::unique_ptr<DeconvolutionAlgorithm> algorithm_;

    DeconvolutionConfig config;

    //memory
    fftw_plan forwardPlan  = nullptr;
    fftw_plan backwardPlan = nullptr;

    fftw_complex *fftwPlanMem = nullptr;

    //multiple psfs
    std::unordered_map<size_t, PSFfftw*> layerPSFMap;
    std::unordered_map<size_t, PSFfftw*> cubePSFMap;

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
    void preprocessPSF(const std::unordered_map<size_t, std::shared_ptr<PSF>>& inputPSFs);
    std::vector<std::vector<cv::Mat>> DeconvolutionProcessor::preprocessChannel(Channel& channel);

    void setPSFShape(const PSF& psf);
    void setImageOriginalShape(const Channel& channel);
    void DeconvolutionProcessor::setCubeShape(
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
    PSFfftw* selectPSFForGridImage(int layerNumber, int cubeNumber) const;
    
    /**
     * @brief Get PSF index for specified layer
     * @param layerNumber Layer number to get PSF for
     * @return Index of PSF in psfs vector, or 0 for default PSF
     */
    PSFfftw* getPSFForLayer(int layerNumber) const;
    
    /**
     * @brief Get PSF index for specified cube
     * @param cubeNumber Cube number to get PSF for
     * @return Index of PSF in psfs vector, or 0 for default PSF
     */
    PSFfftw* getPSFForCube(int cubeNumber) const;

private:
    void deconvolveSingleInplace(int gridIndex, std::vector<std::vector<cv::Mat>>& gridImages);

    bool configured = false;
    // Internal helper functions
    void setupCubeArrangement();
    bool validateImageAndPsfSizes();
    void printConfigurationSummary() const;
    bool setupFFTWPlans();
};