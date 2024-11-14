#pragma once

#include <string>
#include <vector>
#include <complex>
#include <opencv2/core/mat.hpp>
#include <fftw3.h>

namespace UtlFFT {

    // Utility functions
    std::vector<std::vector<cv::Mat>> split3DImageIntoCubes(const std::vector<cv::Mat>& volume, int gridDivision);
    std::vector<cv::Mat> mergeCubesInto3DImage(const std::vector<std::vector<cv::Mat>>& subVolumes, int gridDivision, int originalDepth, int originalHeight, int originalWidth);

    void reorderLayers(fftw_complex* data, int width, int height, int depth);
    void visualizeFFT(fftw_complex* data, int width, int height, int depth);


    // FFT functions
    void convertCVMatVectorToFFTWComplex(const std::vector<cv::Mat>& input, fftw_complex* output, int width, int height, int depth);
    void convertFFTWComplexToCVMatVector(const fftw_complex* input,std::vector<cv::Mat>& output, int width, int height, int depth);

    void padPSF(fftw_complex* psf, int psf_width, int psf_height, int psf_depth, fftw_complex* padded_psf, int width, int height, int depth);

    void forwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth);
    void backwardFFT(fftw_complex* in, fftw_complex* out,int imageDepth, int imageHeight, int imageWidth);

    void octantFourierShift(fftw_complex* data, int width, int height, int depth);
    void inverseQuadrantShift(fftw_complex* data, int width, int height, int depth);
    void quadrantShiftMat(cv::Mat& magI);

    void complexMultiplication(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size);
    void complexDivision(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon);
    void complexAddition(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size);
    void scalarMultiplication(fftw_complex* a, double scalar, fftw_complex* result, int size);
    void complexMultiplicationWithConjugate(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size);
    void complexDivisionStabilized(fftw_complex* a, fftw_complex* b, fftw_complex* result, int size, double epsilon);



        void calculateLaplacianOfPSF(fftw_complex* psf, fftw_complex* laplacian, int width, int height, int depth);

    void normalizeImage(fftw_complex* resultImage, int size, double epsilon);
    void rescaledInverse(fftw_complex* data, double cubeVolume);
    void saveInterimImages(fftw_complex* resultImage, int imageWidth, int imageHeight, int imageDepth, int gridNum, int channel_z, int i);


    void gradientX(fftw_complex* image, fftw_complex* gradX, int width, int height, int depth);
    void gradientY(fftw_complex* image, fftw_complex* gradY, int width, int height, int depth);
    void gradientZ(fftw_complex* image, fftw_complex* gradZ, int width, int height, int depth);
    void computeTV(double lambda, fftw_complex* gx, fftw_complex* gy, fftw_complex* gz, fftw_complex* tv, int width, int height, int depth);
    void normalizeTV(fftw_complex* gradX, fftw_complex* gradY, fftw_complex* gradZ, int width, int height, int depth, double epsilon);



    }