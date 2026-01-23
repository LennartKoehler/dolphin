#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>
#include "backend/ComplexData.h"

namespace UtlFFT {

    // Utility functions
    std::vector<std::vector<cv::Mat>> split3DImageIntoCubes(const std::vector<cv::Mat>& volume, int gridDivision);
    std::vector<cv::Mat> mergeCubesInto3DImage(const std::vector<std::vector<cv::Mat>>& subVolumes, int gridDivision, int originalDepth, int originalHeight, int originalWidth);

    void reorderLayers(complex_t* data, int width, int height, int depth);
    void visualizeFFT(complex_t* data, int width, int height, int depth);

    // FFT functions
    void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, complex_t* output, int width, int height, int depth);
    void convertFFTWComplexToCVMatVector(const complex_t* input,std::vector<cv::Mat>& output, int width, int height, int depth);

    void convertFFTWComplexRealToCVMatVector(const complex_t* input,std::vector<cv::Mat>& output, int width, int height, int depth);
    void convertFFTWComplexImgToCVMatVector(const complex_t* input,std::vector<cv::Mat>& output, int width, int height, int depth);


    void padPSF(complex_t* psf, int psf_width, int psf_height, int psf_depth, complex_t* padded_psf, int width, int height, int depth);

    void forwardFFT(complex_t* in, complex_t* out,int imageDepth, int imageHeight, int imageWidth);
    void backwardFFT(complex_t* in, complex_t* out,int imageDepth, int imageHeight, int imageWidth);

    void octantFourierShift(complex_t* data, int width, int height, int depth);
    void inverseQuadrantShift(complex_t* data, int width, int height, int depth);
    void quadrantShiftMat(cv::Mat& magI);

    void complexMultiplication(complex_t* a, complex_t* b, complex_t* result, int size);
    void complexDivision(complex_t* a, complex_t* b, complex_t* result, int size, double epsilon);
    void complexAddition(complex_t* a, complex_t* b, complex_t* result, int size);
    void scalarMultiplication(complex_t* a, double scalar, complex_t* result, int size);
    void complexMultiplicationWithConjugate(complex_t* a, complex_t* b, complex_t* result, int size);
    void complexDivisionStabilized(complex_t* a, complex_t* b, complex_t* result, int size, double epsilon);

    void calculateLaplacianOfPSF(complex_t* psf, complex_t* laplacian, int width, int height, int depth);

    void normalizeImage(complex_t* resultImage, int size, double epsilon);
    void rescaledInverse(complex_t* data, double cubeVolume);
    void saveInterimImages(complex_t* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i);

    void gradientX(complex_t* image, complex_t* gradX, int width, int height, int depth);
    void gradientY(complex_t* image, complex_t* gradY, int width, int height, int depth);
    void gradientZ(complex_t* image, complex_t* gradZ, int width, int height, int depth);
    void computeTV(double lambda, complex_t* gx, complex_t* gy, complex_t* gz, complex_t* tv, int width, int height, int depth);
    void normalizeTV(complex_t* gradX, complex_t* gradY, complex_t* gradZ, int width, int height, int depth, double epsilon);
}