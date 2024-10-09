#include "BaseDeconvolutionAlgorithm.h"
#include "UtlImage.h"
#include "UtlGrid.h"
#include "UtlFFT.h"
#include <fftw3.h>
#include <opencv2/core.hpp>
#include <iostream>
#include <opencv2/imgproc.hpp>

bool BaseDeconvolutionAlgorithm::preprocess(Channel& channel, PSF& psf) {

        // Find and display global min and max of the data
        double globalMin, globalMax;
        UtlImage::findGlobalMinMax(channel.image.slices, globalMin, globalMax);
        std::cout << "[INFO] Image values min/max: " << globalMin << "/" << globalMax << std::endl;
        double globalMinPsf, globalMaxPsf;
        UtlImage::normalize(psf.image.slices);
        UtlImage::findGlobalMinMax(psf.image.slices, globalMinPsf, globalMaxPsf);
        std::cout << "[INFO] PSF values min/max: " << globalMinPsf << "/" << globalMaxPsf << std::endl;


        int originImageWidth = channel.image.slices[0].cols;
        int originImageHeight = channel.image.slices[0].rows;
        int originImageDepth = channel.image.slices.size();
        int originImageVolume = originImageWidth * originImageHeight * originImageDepth;
        int originPsfWidth = psf.image.slices[0].cols;
        int originPsfHeight = psf.image.slices[0].rows;
        int originPsfDepth = psf.image.slices.size();
        int originPsfVolume = originPsfWidth * originPsfHeight * originPsfDepth;

        //int psfSafetyBorder = 20;//originPsfWidth/2;
        int safetyBorderPsfWidth = psf.image.slices[0].cols+(2*this->psfSafetyBorder);
        int safetyBorderPsfHeight = psf.image.slices[0].rows+(2*this->psfSafetyBorder);
        int safetyBorderPsfDepth = psf.image.slices.size()+(2*this->psfSafetyBorder);
        int safetyBorderPsfVolume = safetyBorderPsfWidth * safetyBorderPsfHeight * safetyBorderPsfDepth;
        int imagePadding = originImageWidth / 2;
        this->cubePadding = this->psfSafetyBorder;
        if(this->cubeSize < 1){
            // Auto function for cubeSize, sets cubeSize to fit PSF
            std::cout << "[INFO] CubeSize fitted to PSF size" << std::endl;
            this->cubeSize = std::min({originPsfWidth, originPsfHeight, originPsfDepth});
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
            std::cout << "[INFO] Gridimage properties: [Depth: " << this->gridImages[0].size() << " Width:" << this->gridImages[0][0].cols << " Height:" << this->gridImages[0][0].rows << " Subimages: " << this->gridImages.size() << "]" << std::endl;

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

        // In-line fftplan for fast ft calculation and inverse
        this->fftwPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * this->cubeVolume);
        this->forwardPlan = fftw_plan_dft_3d(this->cubeDepth, this->cubeHeight, this->cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_FORWARD, FFTW_MEASURE);
        this->backwardPlan = fftw_plan_dft_3d(this->cubeDepth, this->cubeHeight, this->cubeWidth, fftwPlanMem, fftwPlanMem, FFTW_BACKWARD, FFTW_MEASURE);
        fftw_complex *fftwPSFPlanMem = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
        fftw_plan forwardPSFPlan = fftw_plan_dft_3d(originPsfDepth, originPsfHeight, originPsfWidth, fftwPSFPlanMem, fftwPSFPlanMem, FFTW_FORWARD, FFTW_MEASURE);

        // Fourier Transformation of PSF
        std::cout << "[STATUS] Performing Fourier Transform on PSF..." << std::endl;
        fftw_complex *h = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * originPsfVolume);
        UtlFFT::convertCVMatVectorToFFTWComplex(psf.image.slices, h, originPsfWidth, originPsfHeight, originPsfDepth);
        fftw_execute_dft(forwardPSFPlan, h, h);
        UtlFFT::octantFourierShift(h, originPsfWidth, originPsfHeight, originPsfDepth);

        std::cout << "[STATUS] Padding PSF..." << std::endl;
        // Pad the PSF to the size of the image
        this->paddedH = (fftw_complex *) fftw_malloc(sizeof(fftw_complex) * safetyBorderPsfVolume);
        UtlFFT::padPSF(h, originPsfWidth, originPsfHeight, originPsfDepth, paddedH, safetyBorderPsfWidth, safetyBorderPsfHeight, safetyBorderPsfDepth);
        // Free FFTW resources for PSF
        fftw_free(h);
        fftw_free(fftwPSFPlanMem);
        fftw_destroy_plan(forwardPSFPlan);

    return true;
}
bool BaseDeconvolutionAlgorithm::postprocess(Hyperstack& data, double epsilon){
    if(this->grid){
        UtlGrid::cropCubePadding(this->gridImages, this->cubePadding);
        this->cubeWidth = this->cubeSize;
        this->cubeHeight = this->cubeSize;
        this->cubeDepth = this->cubeSize;
        this->cubeVolume = this->cubeWidth * this->cubeHeight * this->cubeDepth;
        std::cout << " " << std::endl;
        std::cout << "[STATUS] Merging Grid back to Image..." << std::endl;
        this->mergedVolume = UtlGrid::mergeCubes(this->gridImages, data.metaData.imageWidth, data.metaData.imageLength, data.metaData.slices, this->cubeSize);
        std::cout << "[INFO] Image size: " << this->mergedVolume[0].rows << "x" << this->mergedVolume[0].cols << "x" << this->mergedVolume.size()<< std::endl;
    }else{
        this->mergedVolume = this->gridImages[0];
    }

    for (auto& slice : this->mergedVolume) {
        //TODOi
        cv::threshold(slice, slice, epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter 0 auf 0 setzen
        //cv::normalize(slice, slice, 0, 1, cv::NORM_MINMAX);
    }

    // Global normalization of the merged volume
    double global_max_val= 0.0;
    double global_min_val = MAXFLOAT;
    int j = 0;
    for (const auto& slice : this->mergedVolume) {
        double min_val, max_val;
        cv::minMaxLoc(slice, &min_val, &max_val);
        global_max_val = std::max(global_max_val, max_val);
        global_min_val = std::min(global_min_val, min_val);

    }

    for (auto& slice : this->mergedVolume) {
        slice.convertTo(slice, CV_32F, 1.0 / (global_max_val - global_min_val), -global_min_val * (1 / (global_max_val - global_min_val)));  // Add epsilon to avoid division by zero
        // cv::threshold(slice, slice, this->epsilon, 0.0, cv::THRESH_TOZERO); // Werte unter 0 auf 0 setzen
    }

    return true;

}
void BaseDeconvolutionAlgorithm::cleanup() {
    // Free FFTW resources for the current channel
    if (this->paddedH) {
        fftw_free(this->paddedH);
        this->paddedH = nullptr;
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


