#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>
#include "complexType.h"

namespace UtlFFT {

    // Utility functions
    std::vector<std::vector<cv::Mat>> split3DImageIntoCubes(const std::vector<cv::Mat>& volume, int gridDivision);
    std::vector<cv::Mat> mergeCubesInto3DImage(const std::vector<std::vector<cv::Mat>>& subVolumes, int gridDivision, int originalDepth, int originalHeight, int originalWidth);

    void reorderLayers(complex* data, int width, int height, int depth);
    void visualizeFFT(complex* data, int width, int height, int depth);

    // FFT functions
    void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, complex* output, int width, int height, int depth);
    void convertFFTWComplexToCVMatVector(const complex* input,std::vector<cv::Mat>& output, int width, int height, int depth);

    void convertFFTWComplexRealToCVMatVector(const complex* input,std::vector<cv::Mat>& output, int width, int height, int depth);
    void convertFFTWComplexImgToCVMatVector(const complex* input,std::vector<cv::Mat>& output, int width, int height, int depth);


    void padPSF(complex* psf, int psf_width, int psf_height, int psf_depth, complex* padded_psf, int width, int height, int depth);

    void forwardFFT(complex* in, complex* out,int imageDepth, int imageHeight, int imageWidth);
    void backwardFFT(complex* in, complex* out,int imageDepth, int imageHeight, int imageWidth);

    void octantFourierShift(complex* data, int width, int height, int depth);
    void inverseQuadrantShift(complex* data, int width, int height, int depth);
    void quadrantShiftMat(cv::Mat& magI);

    void complexMultiplication(complex* a, complex* b, complex* result, int size);
    void complexDivision(complex* a, complex* b, complex* result, int size, double epsilon);
    void complexAddition(complex* a, complex* b, complex* result, int size);
    void scalarMultiplication(complex* a, double scalar, complex* result, int size);
    void complexMultiplicationWithConjugate(complex* a, complex* b, complex* result, int size);
    void complexDivisionStabilized(complex* a, complex* b, complex* result, int size, double epsilon);

    void calculateLaplacianOfPSF(complex* psf, complex* laplacian, int width, int height, int depth);

    void normalizeImage(complex* resultImage, int size, double epsilon);
    void rescaledInverse(complex* data, double cubeVolume);
    void saveInterimImages(complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i);

    void gradientX(complex* image, complex* gradX, int width, int height, int depth);
    void gradientY(complex* image, complex* gradY, int width, int height, int depth);
    void gradientZ(complex* image, complex* gradZ, int width, int height, int depth);
    void computeTV(double lambda, complex* gx, complex* gy, complex* gz, complex* tv, int width, int height, int depth);
    void normalizeTV(complex* gradX, complex* gradY, complex* gradZ, int width, int height, int depth, double epsilon);
}