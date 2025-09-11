#pragma once
#include <string>
#include "../DeconvolutionConfig.h"
#include "HyperstackImage.h"
#include "psf/PSF.h"
#include "BaseDeconvolutionAlgorithm.h"

#ifdef CUDA_AVAILABLE
#include <cufftw.h>
#else
#include <fftw3.h>
#endif

typedef size_t PSFIndex;
typedef fftw_complex PSFfftw;



struct CubeMetaData{
    int originPsfWidth;
    int originPsfHeight;
    int originPsfDepth;
    int originPsfVolume;

    // Calculate safety border for PSF padding
    int cubeWidth;
    int cubeHeight;
    int cubeDepth;
    int cubeVolume;
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
class BaseDeconvolutionAlgorithmDerived : public BaseDeconvolutionAlgorithm {
public:
    Hyperstack run(Hyperstack& data, std::vector<PSF>& psfs);
    
    virtual ~BaseDeconvolutionAlgorithmDerived() { cleanup(); }
    
    // Override base virtual methods to separate concerns
    virtual void configure(DeconvolutionConfig config);
    virtual void preprocessPSFS(const std::vector<PSF>& psfs,
        const std::unordered_map<size_t, PSFIndex>& layerPSFMap,
        const std::unordered_map<size_t, PSFIndex>& cubePSFMap);
    virtual void algorithm(Hyperstack& data, int channel_num, fftw_complex* H, fftw_complex* g, fftw_complex* f) override = 0;

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
    // Common configuration members - directly accessible by derived classes
    double epsilon;      // Minimum threshold for values
    bool time;           // Whether to measure and display timing
    bool saveSubimages;   // Whether to save subimage results
    bool grid;           // Whether to use grid processing
    std::string gpu;    // GPU API selection ("cuda", "", "opencl")
    // Grid processing parameters
    int borderType;         // Border type for image extension
    int psfSafetyBorder;    // Safety border around PSF
    int cubeSize;           // Size of cubic subimages
    //multiple psfs
    std::unordered_map<size_t, PSFfftw*> layerPSFMap;
    std::unordered_map<size_t, PSFfftw*> cubePSFMap;
    CubeMetaData cubeMetaData;
    // std::vector<PSFfftw*> paddedHs;
    fftw_complex *fftwPlanMem = nullptr;
 
    // Helper functions that don't depend on execution backend
    
    /**
     * @brief Configure grid processing parameters
     * @param cubeSize Size of cubic subimages
     */
    void configureGridProcessing(int cubeSize);
    
    /**
     * @brief Prepare PSF for processing - performs FFT and padding
     * @param psfs List of PSFs to prepare
     * @return true if PSF preparation succeeded, false otherwise
     */
    std::vector<PSFfftw*> preparePSFs(const std::vector<PSF>& psfs);
    void configureLayerMap(const std::vector<PSFfftw*>& psfs, const std::unordered_map<size_t, PSFIndex> layerPSFMap);
    void configureCubeMap(const std::vector<PSFfftw*>& psfs, const std::unordered_map<size_t, PSFIndex> cubePSFMap);

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
    // Internal helper functions
    void setupCubeArrangement();
    bool validateImageAndPsfSizes();
    void printConfigurationSummary() const;
    bool setupFFTWPlans();
};