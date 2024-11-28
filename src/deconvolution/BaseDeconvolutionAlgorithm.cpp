#include "BaseDeconvolutionAlgorithm.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include "UtlFFT.h"
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <omp.h>
#ifdef CUDA_AVAILABLE
#include <cufftw.h>
#include <operations.h>
#include <utl.h>
#else
#include <fftw3.h>
#endif


bool BaseDeconvolutionAlgorithm::preprocess(Channel& channel, std::vector<PSF>& psfs) {
        // Find and display global min and max of the data
        double globalMin, globalMax;
        UtlImage::findGlobalMinMax(channel.image.slices, globalMin, globalMax);
        std::cout << "[INFO] Image values min/max: " << globalMin << "/" << globalMax << std::endl;
        int psfcount = 1;
        for (PSF psf : psfs) {
            double globalMinPsf, globalMaxPsf;
            UtlImage::findGlobalMinMax(psf.image.slices, globalMinPsf, globalMaxPsf);
            std::cout << "[INFO] PSF" << "_" << psfcount<<" values min/max: " << globalMinPsf << "/" << globalMaxPsf << std::endl;
            if(!this->secondPSF){
                break;
            }
            psfcount++;
        }

        int originImageWidth = channel.image.slices[0].cols;
        this->originalImageWidth = originImageWidth;
        int originImageHeight = channel.image.slices[0].rows;
        this->originalImageHeight = originImageHeight;
        int originImageDepth = channel.image.slices.size();
        this->originalImageDepth = originImageDepth;
        int originImageVolume = originImageWidth * originImageHeight * originImageDepth;
        int originPsfWidth = psfs[0].image.slices[0].cols;
        int originPsfHeight = psfs[0].image.slices[0].rows;
        int originPsfDepth = psfs[0].image.slices.size();
        int originPsfVolume = originPsfWidth * originPsfHeight * originPsfDepth;

        //int psfSafetyBorder = 20;//originPsfWidth/2;
        int safetyBorderPsfWidth = psfs[0].image.slices[0].cols+(2*this->psfSafetyBorder);
        int safetyBorderPsfHeight = psfs[0].image.slices[0].rows+(2*this->psfSafetyBorder);
        int safetyBorderPsfDepth = psfs[0].image.slices.size()+(2*this->psfSafetyBorder);
        int safetyBorderPsfVolume = safetyBorderPsfWidth * safetyBorderPsfHeight * safetyBorderPsfDepth;
        int imagePadding = originImageWidth / 2;
        this->cubePadding = this->psfSafetyBorder;
        if(this->cubeSize < 1){
            // Auto function for cubeSize, sets cubeSize to fit PSF
            std::cout << "[INFO] CubeSize fitted to PSF size" << std::endl;
            this->cubeSize = std::max({originPsfWidth, originPsfHeight, originPsfDepth});
        }

        if(safetyBorderPsfWidth < this->cubeSize){
            this->cubePadding = 10;
            std::cout << "[INFO] PSF with safety border smaller than cubeSize" << std::endl;
        }
        if(this->cubeSize+2*this->cubePadding < safetyBorderPsfWidth){
            this->cubePadding = (safetyBorderPsfWidth-this->cubeSize)/2;
            //std::cout <<  "[INFO] cubeSize smaller than PSF with safety border" << std::endl;
        }
        if(!this->grid){
            std::cout << "[INFO] Processing without grid" << std::endl;
            this->gridImages.push_back(channel.image.slices);
            this->cubeWidth = channel.image.slices[0].cols;
            this->cubeHeight = channel.image.slices[0].rows;
            this->cubeDepth = channel.image.slices.size();
            this->cubeVolume = cubeWidth * cubeHeight * cubeDepth;

            safetyBorderPsfWidth = this->cubeWidth;
            safetyBorderPsfHeight = this->cubeHeight;
            safetyBorderPsfDepth = this->cubeDepth;
            safetyBorderPsfVolume = this->cubeVolume;

        }else {

            if(this->psfSafetyBorder < 1){
                std::cerr << "[ERROR] CubeSize should be greater than 1 and PsfSafetyBorder should be greater than 0" << std::endl;
                return false;
            }

            UtlGrid::extendImage(channel.image.slices, imagePadding, this->borderType);

            this->gridImages = UtlGrid::splitWithCubePadding(channel.image.slices, this->cubeSize, imagePadding, this->cubePadding);
            std::cout << "[INFO] Actual cubeSize: " << this->cubeSize << "px" << std::endl;
            std::cout << "[INFO] GridImage(with extentsion) properties: [Depth: " << this->gridImages[0].size() << " Width:" << this->gridImages[0][0].cols << " Height:" << this->gridImages[0][0].rows << " Subimages: " << this->gridImages.size() << "]" << std::endl;

            if((this->cubeSize + 2*this->cubePadding) != this->gridImages[0][0].cols){
                std::cerr << "[ERROR] CubeSize doesnt match with actual CubeSize: " << this->gridImages[0][0].cols << " (should be: " << (this->cubeSize + 2*this->cubePadding) << ")" << std::endl;
                return false;
            }

            this->cubeWidth = (this->cubeSize + 2*this->cubePadding);
            this->cubeHeight = (this->cubeSize + 2 * this->cubePadding);
            this->cubeDepth = (this->cubeSize + 2*this->cubePadding);
            this->cubeVolume = this->cubeWidth * this->cubeHeight * this->cubeDepth;

            if(this->cubeWidth != this->psfSafetyBorder){
                if(this->cubeWidth > this->psfSafetyBorder){
                    safetyBorderPsfWidth = this->cubeWidth;
                    safetyBorderPsfHeight = this->cubeHeight;
                    safetyBorderPsfDepth = this->cubeDepth;
                    safetyBorderPsfVolume = this->cubeVolume;
                }
            }

        }
        if(this->cubeSize < originPsfWidth){
            std::cout << "[WARNING] PSF is larger than image/cube" << std::endl;
        }

        std::cout << "[STATUS] Creating fftw plans..." << std::endl;
        // In-line fftplan for fast ft calculation and inverse
        this->fftwPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        this->forwardPlan = fftw_plan_dft_3d(this->cubeDepth, this->cubeHeight, this->cubeWidth, this->fftwPlanMem, this->fftwPlanMem, FFTW_FORWARD, FFTW_MEASURE);
        this->backwardPlan = fftw_plan_dft_3d(this->cubeDepth, this->cubeHeight, this->cubeWidth, this->fftwPlanMem, this->fftwPlanMem, FFTW_BACKWARD, FFTW_MEASURE);
        fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
        fftw_plan forwardPSFPlan = fftw_plan_dft_3d(originPsfDepth, originPsfHeight, originPsfWidth, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);

        // Fourier Transformation of PSF
        std::cout << "[STATUS] Performing Fourier Transform on PSF..." << std::endl;
        fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
        UtlFFT::convertCVMatVectorToFFTWComplex(psfs[0].image.slices, h, originPsfWidth, originPsfHeight, originPsfDepth);
        fftw_execute_dft(forwardPSFPlan, h, h);

        std::cout << "[STATUS] Padding PSF..." << std::endl;
        // Pad the PSF to the size of the image
        this->paddedH = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
        UtlFFT::padPSF(h, originPsfWidth, originPsfHeight, originPsfDepth, this->paddedH, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);

        if(this->secondPSF){
            if(psfs.size() < 2) {
                std::cerr << "[ERROR] Only one PSF loaded" << std::endl;
                return false;
            }
            //second PSF
            std::cout << "[STATUS] Performing Fourier Transform on PSF_2..." << std::endl;
            fftw_complex *h_2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
            UtlFFT::convertCVMatVectorToFFTWComplex(psfs[1].image.slices, h_2, originPsfWidth, originPsfHeight, originPsfDepth);
            fftw_execute_dft(forwardPSFPlan, h_2, h_2);
            std::cout << "[STATUS] Padding PSF_2..." << std::endl;
            // Pad the PSF to the size of the image
            this->paddedH_2 = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
            UtlFFT::padPSF(h_2, originPsfWidth, originPsfHeight, originPsfDepth, this->paddedH_2, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
            fftw_free(h_2);
        }

#ifdef CUDA_AVAILABLE
    if(this->gpu == "cuda") {
        //INFO safetyBorderPsfVolume = this->cubeWidth* this->cubeHeight* this->cubeDepth
        cudaMalloc((void**)&this->d_paddedH, safetyBorderPsfVolume * sizeof(fftw_complex));
        copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, this->d_paddedH, this->paddedH);
        if(this->secondPSF) {
            cudaMalloc((void**)&this->d_paddedH_2, safetyBorderPsfVolume * sizeof(fftw_complex));
            copyDataFromHostToDevice(this->cubeWidth, this->cubeHeight, this->cubeDepth, this->d_paddedH_2, this->paddedH_2);
        }
    }
#else
#endif

        // Free FFTW resources for PSF
        fftw_free(h);
        fftw_free(fftwPSFPlanMem);
        fftw_destroy_plan(forwardPSFPlan);


    return true;
}
bool BaseDeconvolutionAlgorithm::postprocess(double epsilon){
    if(this->gridImages.empty()){
        std::cerr << "[ERROR] No grid images(cubes) processed" << std::endl;
        return false;
    }
    if(this->grid){
        //TODO no effect
        //UtlGrid::adjustCubeOverlap(this->gridImages,this->cubePadding);

        UtlGrid::cropCubePadding(this->gridImages, this->cubePadding);
        std::cout << " " << std::endl;
        std::cout << "[STATUS] Merging Grid back to Image..." << std::endl;
        this->mergedVolume = UtlGrid::mergeCubes(this->gridImages, this->originalImageWidth, this->originalImageHeight, this->originalImageDepth, this->cubeSize);
        std::cout << "[INFO] Image size: " << this->mergedVolume[0].rows << "x" << this->mergedVolume[0].cols << "x" << this->mergedVolume.size()<< std::endl;
    }else{
        this->mergedVolume = this->gridImages[0];
    }


    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    int j = 0;
    for (const auto& slice : this->mergedVolume) {
        cv::threshold(slice, slice, 0, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);
    }

    for (auto& slice : this->mergedVolume) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));  // Add epsilon to avoid division by zero
        cv::threshold(slice, slice, epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter epsilon auf 0 setzen
    }


    return true;

}
void BaseDeconvolutionAlgorithm::cleanup() {
    // Free FFTW resources for the current channel
    if (this->paddedH) {
        fftw_free(this->paddedH);
        this->paddedH = nullptr;
#ifdef CUDA_AVAILABLE
        if(this->gpu == "cuda") {
            cudaFree(this->d_paddedH);
            this->d_paddedH = nullptr;
        }
#endif
    }
    if (this->paddedH_2) {
        fftw_free(this->paddedH_2);
        this->paddedH_2 = nullptr;
#ifdef CUDA_AVAILABLE
        if(this->gpu == "cuda") {
            cudaFree(this->d_paddedH_2);
            this->d_paddedH_2 = nullptr;
        }
#endif
    }
    if (this->fftwPlanMem) {
        fftw_free(this->fftwPlanMem);
        this->fftwPlanMem = nullptr;
    }
    if (this->forwardPlan) {
        fftw_destroy_plan(this->forwardPlan);
        this->forwardPlan = nullptr;
    }
    if (this->backwardPlan) {
        fftw_destroy_plan(this->backwardPlan);
        this->backwardPlan = nullptr;
    }
    // Clear the subimage vector
    this->gridImages.clear();
}
Hyperstack BaseDeconvolutionAlgorithm::deconvolve(Hyperstack &data, std::vector<PSF> &psfs) {
    // Create a copy of the input data
    Hyperstack deconvHyperstack{data};

#ifdef CUDA_AVAILABLE
    if(this->gpu == "cuda") {
        std::cout << "[INFO] Using CUDA" << std::endl;
    }
#else
    if(this->gpu == "cuda") {
        std::cerr << "[ERROR] CUDA is not available" << std::endl;
        return deconvHyperstack;
    }
        // Init threads for FFTW
        if(fftw_init_threads() > 0){
            std::cout << "[STATUS] FFTW init threads" << std::endl;
            fftw_plan_with_nthreads(omp_get_max_threads());
            std::cout << "[INFO] Available threads: " << omp_get_max_threads() << std::endl;
            fftw_make_planner_thread_safe();
        }
#endif

    // Deconvolve every channel
    int channel_z = 0;
    for (auto& channel : data.channels) {
        if(preprocess(channel, psfs)){
            std::cout << "[STATUS] Preprocessing channel " << channel_z + 1 << " finished" << std::endl;
        }else{
            std::cerr << "[ERROR] Preprocessing channel " << channel_z + 1 << " failed" << std::endl;
            return deconvHyperstack;
        }

        // Debug
        //std::cout << originPsfWidth << " " << originPsfHeight << " " << originPsfDepth << std::endl;
        //std::cout << safetyBorderPsfWidth << " " << safetyBorderPsfHeight << " " << safetyBorderPsfDepth << std::endl;
        //std::cout << cubeWidth << " " << cubeHeight << " " << cubeDepth << std::endl;

        // Prepare info of cube arrangement
        std::cout << "[STATUS] Running Deconvolution..." << std::endl;
        this->cubesPerX = static_cast<int>(std::ceil(static_cast<double>(this->originalImageWidth) / this->cubeSize));
        this->cubesPerY = static_cast<int>(std::ceil(static_cast<double>(this->originalImageHeight) / this->cubeSize));
        this->cubesPerZ = static_cast<int>(std::ceil(static_cast<double>(this->originalImageDepth) / this->cubeSize));
        this->cubesPerLayer = cubesPerX * cubesPerY;
        std::cout << "[INFO] Cubes per Layer(" << cubesPerZ<< "):" << cubesPerX << "x" << cubesPerY << " (" << cubesPerLayer << ")" << std::endl;

        // Parallelization of grid for
// Using static scheduling because the execution time for each iteration is similar, which reduces overhead costs by minimizing task assignment.
#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < this->gridImages.size(); ++i) {
            // PSF
            // H points to an existing PSF (paddedH or paddedH_2) and should not be freed here as it is not allocated with fftw_malloc.
            fftw_complex* H = nullptr;
            // Observed image
            fftw_complex* g = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
            // Result image
            fftw_complex* f = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);

            if(this->secondPSF){
                // Check if second PSF has to be applied
                int currentCubeLayer = static_cast<int>(std::ceil(static_cast<double>((i+1)) / this->cubesPerLayer));
                auto useSecondPsfForThisLayer = std::find(secondpsflayers.begin(), secondpsflayers.end(), currentCubeLayer);
                auto useSecondPsfForThisCube = std::find(secondpsfcubes.begin(), secondpsfcubes.end(), i+1);
                // Load the correct PSF
                if (useSecondPsfForThisLayer != secondpsflayers.end() ||  useSecondPsfForThisCube != secondpsfcubes.end()) {
                    //std::cout << "[DEBUG] second PSF at "<< i+1 << std::endl;
#ifdef CUDA_AVAILABLE
                    if(this->gpu == "cuda") {H = this->d_paddedH_2;}else {H = this->paddedH_2;}
#else
                    H = this->paddedH_2;
#endif
                } else {
                    //std::cout << "[DEBUG] first PSF at "<< i+1 << std::endl;
#ifdef CUDA_AVAILABLE
                    if(this->gpu == "cuda") {H = this->d_paddedH;}else {H = this->paddedH;}
#else
                    H = this->paddedH;
#endif
                }
            }else{
#ifdef CUDA_AVAILABLE
                if(this->gpu == "cuda") {H = this->d_paddedH;}else {H = this->paddedH;}
#else
                H = this->paddedH;
#endif
            }

            // Convert image to fftcomplex
            UtlFFT::convertCVMatVectorToFFTWComplex(this->gridImages[i], g, this->cubeWidth, this->cubeHeight, this->cubeDepth);

            std::cout << "\r[STATUS] Channel: " << channel_z + 1 << "/" << data.channels.size() << " GridImage: "
                      << this->totalGridNum << "/" << this->gridImages.size() << " ";
            // Methode overridden in specific algorithm class
            if (!(UtlImage::isValidForFloat(g, this->cubeVolume))) {
                std::cout << "[WARNING LOL] Value fftwPlanMem fftwcomplex(double) is smaller than float" << std::endl;
            }

            algorithm(data, channel_z, H, g, f);

            // Convert the result FFTW complex array back to OpenCV Mat vector
            UtlFFT::convertFFTWComplexToCVMatVector(f, this->gridImages[i], this->cubeWidth, this->cubeHeight, this->cubeDepth);

            this->totalGridNum++;
            fftw_free(g);
            fftw_free(f);
            std::flush(std::cout);

        }
        // this->girdImages of BaseDeconvolutionAlgorithm deconvolution complete

        // Debug ouptut
        // 1. size in metadata, 2. size of extendes channel image, 3.original read in image size
        //std::cout << " " << std::endl;
        //std::cout << data.metaData.imageWidth << std::endl;
        //std::cout << channel.image.slices[0].cols << std::endl;
        //std::cout << this->originalImageWidth << std::endl;
        //std::cout << " " << std::endl;
        //std::cout << data.metaData.imageLength << std::endl;
        //std::cout << channel.image.slices[0].rows << std::endl;
        //std::cout << this->originalImageHeight << std::endl;
        //std::cout << " " << std::endl;
        //std::cout << data.metaData.slices << std::endl;
        //std::cout << channel.image.slices.size() << std::endl;
        //std::cout << this->originalImageDepth << std::endl;

        if(postprocess(this->epsilon)){
            std::cout << "[STATUS] Postprocessing channel " << channel_z + 1 << " finished" << std::endl;
        }else{
            std::cerr << "[ERROR] Postprocessing channel " << channel_z + 1 << " failed" << std::endl;
            return deconvHyperstack;
        }

        // Save the result
        std::cout << "[STATUS] Saving result of channel " << channel_z + 1 << std::endl;
        Image3D deconvolutedImage;
        deconvolutedImage.slices = this->mergedVolume;
        deconvHyperstack.channels[channel.id].image = deconvolutedImage;
        channel_z++;
        this->mergedVolume.clear();
    }

    std::cout << "[STATUS] Deconvolution complete" << std::endl;
    return deconvHyperstack;
}


