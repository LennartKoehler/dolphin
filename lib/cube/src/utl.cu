#include "utl.h"
#include "kernels.h"
#include <cuComplex.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>


// Print information
void printDeviceProperties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0); // 0 is ID of GPU

    std::cout << "Name: " << prop.name << std::endl;
    std::cout << "CUDA Version: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    std::cout << "Max block dimensions: " << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x " << prop.maxThreadsDim[2] << std::endl;
    std::cout << "Max grid dimensions: " << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x " << prop.maxGridSize[2] << std::endl;
    std::cout << "Global memory size: " << prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024 << " KB" << std::endl;
    std::cout << "Memory latency: " << prop.memoryBusWidth << " bits" << std::endl;
    std::cout << "Multiprocessor count: " << prop.multiProcessorCount << std::endl;
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << std::endl;
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << " " << std::endl;
}
void printFirstElem(fftw_complex* mat) {
    std::cout <<"[CHECK]["<< mat[0][0] << "+" << mat[0][1] <<"]" << std::endl;
}
void printSpecificElem(fftw_complex* mat, int index) {
    std::cout <<"[CHECK]["<< mat[index][0] << "+" << mat[index][1] <<"]" << std::endl;
}
void printRandomElem(fftw_complex* mat, int N) {
    int size = N * N * N;
    int randomIndex = rand() % size;
    std::cout <<"[CHECK]["<< mat[randomIndex][0] << "+" << mat[randomIndex][1] <<"]" << std::endl;
}


// Check Mat
void checkUniformity(fftw_complex* mat, int N) {
    // Take the first element as a reference
    double *reference = mat[0];
    int countEqual = 0;
    int countDifferent = 0;
    double maxDeviation = 0.0;
    double totalDeviation = 0.0;
    int totalElements = N * N * N;

    for (int i = 0; i < totalElements; i++) {
        double realDiff = std::fabs(mat[i][0] - reference[0]);
        double imagDiff = std::fabs(mat[i][1] - reference[1]);

        if (realDiff == 0 && imagDiff == 0) {
            countEqual++;
        } else {
            countDifferent++;

            // Calculate the largest deviation for each element
            double deviation = std::sqrt(realDiff * realDiff + imagDiff * imagDiff);
            maxDeviation = std::max(maxDeviation, deviation);

            totalDeviation += deviation; // Total deviation for average
        }
    }

    if (countDifferent == 0) {
        std::cout << "[CHECK] All elements are equal to the reference value: "
                  << reference[0] << " + " << reference[1] << "i" << std::endl;
        std::cout << "[CHECK] Number of elements (" << N << "x" << N << "x" << N << "): "
                  << countEqual << std::endl;
    } else {
        std::cout << "[CHECK] Matrix is not uniform." << std::endl;
        std::cout << "[CHECK] Number of matching elements: " << countEqual << std::endl;
        std::cout << "[CHECK] Number of different elements: " << countDifferent << std::endl;

        double avgDeviation = totalDeviation / countDifferent;
        std::cout << "[CHECK] Maximum deviation: " << maxDeviation << std::endl;
        std::cout << "[CHECK] Average deviation: " << avgDeviation << std::endl;
    }
}
void displayHeatmap(const fftw_complex* data, int N) {
    int currentSlice = 0;  // Beginne mit dem ersten Slice

    // Create an OpenCV window (this remains open throughout the entire process)
    cv::namedWindow("Heatmap Slice Viewer", cv::WINDOW_NORMAL);

    while (true) {
        // Create a 2D array for the current slice
        cv::Mat heatmap(N, N, CV_32F);  // Heatmap with 32-bit float

        // Fill the heatmap with the values of the current slice
        for (int y = 0; y < N; ++y) {
            for (int x = 0; x < N; ++x) {
                int idx = currentSlice * N * N + y * N + x;  // indexes for slices
                heatmap.at<float>(y, x) = std::abs(data[idx][0]);  // real as example
            }
        }

        // Normalize the heatmap and convert it to 8-bit for display
        cv::normalize(heatmap, heatmap, 0, 255, cv::NORM_MINMAX);
        heatmap.convertTo(heatmap, CV_8U);

        // Create the title for the window (displays the slice) and show
        std::string title = "Slice " + std::to_string(currentSlice + 1) + " von " + std::to_string(N);
        cv::imshow("Heatmap Slice Viewer", heatmap);

        // Wait for a key press (waiting for 0ms means it waits for input immediately)
        int key = cv::waitKey(0);

        // If 'ENTER' is pressed (Code 13), go to the next slice
        if (key == 13) {
            currentSlice = (currentSlice + 1) % N;  // Zykliere über alle Slices
        }
        // If 'ESC' is pressed (Code 27), break the loop and exit the program
        else if (key == 27) {
            break;
        }
    }
    cv::destroyWindow("Heatmap Slice Viewer");
}
bool checkOctantFourierShift(int N, fftw_complex* original, fftw_complex* shifted) {
    int width = N, height = N, depth = N;
    int halfWidth = width / 2;
    int halfHeight = height / 2;
    int halfDepth = depth / 2;

    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                // Current index in the shifted array
                int idxShifted = z * height * width + y * width + x;

                // Calculate the origin from which this element was shifted
                int idxOriginal = ((z + halfDepth) % depth) * height * width +
                                  ((y + halfHeight) % height) * width +
                                  ((x + halfWidth) % width);

                // Check if the values at this index match
                if (std::abs(original[idxOriginal][0] - shifted[idxShifted][0]) > 1e-6 ||
                    std::abs(original[idxOriginal][1] - shifted[idxShifted][1]) > 1e-6) {
                    std::cout << "[ERROR] Fourier Shift not correct at index ("
                              << x << ", " << y << ", " << z << ")\n";
                    return false;
                    }
            }
        }
    }

    std::cout << "Octant Fourier shift successfully validated!\n";
    return true;
}


// Mat initialization
void createFftwUniformMat(int N, fftw_complex* mat){
#pragma omp parallel for
    for (int i = 0; i < N*N*N; i++) {
        mat[i][0] = 2.0f;  // real
        mat[i][1] = 0.0f;  // img
    }
}
void createFftwRandomMat(int N, fftw_complex* mat) {
    // Initialize the random number generator with the current time
    std::srand(static_cast<unsigned int>(std::time(0)));

    // Fill the matrix with random values in the range [0, 1] in steps of 0.001
#pragma omp parallel for
    for (int i = 0; i < N * N * N; i++) {
        // Generate a random number in the range [0, 1000] and divide by 1000 to get the range [0, 1]
        double randReal = (std::rand() % 1001) / 1000.0;
        double randImag = (std::rand() % 1001) / 1000.0;

        mat[i][0] = randReal;  // real
        mat[i][1] = randImag;  // img
    }
}
void createFftwSphereMat(int N, fftw_complex* mat) {
    // Determine the center of the matrix
    int center = N / 2;
    int radius = N / 2;

#pragma omp parallel for
    for (int i = 0; i < N * N * N; i++) {
        // Calculate coordinates
        int z = i / (N * N);
        int y = (i % (N * N)) / N;
        int x = i % N;

        // Calculate the distance of the point (x, y, z) from the center of the matrix
        int distSq = (x - center) * (x - center) + (y - center) * (y - center) + (z - center) * (z - center);

        // If the distance is within the radius, set the value to 1, otherwise set it to 0
        if (distSq <= radius * radius) {
            mat[i][0] = 1.0f;  // real
            mat[i][1] = 0.0f;  // img
        } else {
            mat[i][0] = 0.0f;  // real
            mat[i][1] = 0.0f;  // img
        }
    }
}


// Copying fftw_complex datatype to GPU
void copyDataFromHostToDevice(fftw_complex* dest, fftw_complex* src, int matrixSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dest, src, matrixSize, cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<matrixSize<<"B] Copy Data from Host to Device" <<  std::endl;
}
void copyDataFromDeviceToHost(fftw_complex* dest, fftw_complex* src, int matrixSize) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cudaMemcpy(dest, src, matrixSize, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
    }

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<matrixSize<<"B] Copy Data from Device to Host" << std::endl;
}


// Conversions
void convertFftwToCuComplexOnDevice(fftw_complex* fftwArr, cuComplex* cuArr, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    fftw_complex* fftwArrDevice;
    cudaMalloc(&fftwArrDevice, N * N * N * sizeof(fftw_complex));

    copyDataFromHostToDevice(fftwArrDevice, fftwArr, N * N * N * sizeof(fftw_complex));

    // Kernel dimension 1D, because 3D matrix stored in 1D array, just copying values at [i]
    int numElements = N * N * N;
    int blockSize = 1024;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    fftwToCuComplexKernelGlobal<<<numBlocks, blockSize>>>(cuArr, fftwArrDevice, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at convertFFTWToCuComplex: " << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(fftwArrDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<N * N * N * sizeof(fftw_complex)<<"B] Converting (inkl. copy) fftw_complex to cuComplex" << std::endl;
}
void convertFftwToCufftComplexOnDevice(fftw_complex* fftwArr, cufftComplex* cuArr, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    fftw_complex* fftwArrDevice;
    cudaMalloc(&fftwArrDevice, N * N * N * sizeof(fftw_complex));

    copyDataFromHostToDevice(fftwArrDevice, fftwArr, N * N * N * sizeof(fftw_complex));

    // Kernel dimension 1D, because 3D matrix stored in 1D array, just copying values at [i]
    int numElements = N * N * N;
    int blockSize = 1024;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    fftwToCuComplexKernelGlobal<<<numBlocks, blockSize>>>(cuArr, fftwArrDevice, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at convertFFTWToCuComplex: " << cudaGetErrorString(err) << std::endl;
    }
    cudaFree(fftwArrDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<N * N * N * sizeof(fftw_complex)<<"B] Converting (inkl. copy) fftw_complex to cufftComplex" << std::endl;
}
void convertCuToFftwComplexOnHost(fftw_complex* fftwArr, cuComplex* cuArr, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    fftw_complex* fftwArrDevice;
    cudaMalloc(&fftwArrDevice, N * N * N * sizeof(fftw_complex));

    int numElements = N * N * N;
    // Kernel dimension 1D, because 3D matrix stored in 1D array, just copying values at [i]
    int blockSize = 1024;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    cuToFftwComplexKernelGlobal<<<numBlocks, blockSize>>>(fftwArrDevice, cuArr, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at convertCuComplexToFFTW: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(fftwArr, fftwArrDevice, numElements * sizeof(fftw_complex), cudaMemcpyDeviceToHost);
    cudaFree(fftwArrDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<N * N * N * sizeof(fftw_complex)<<"B] Converting (inkl. copy) cuComplex to fftw_complex" << std::endl;
}
void convertCufftToFftwComplexOnHost(fftw_complex* fftwArr, cufftComplex* cuArr, int N) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    fftw_complex* fftwArrDevice;
    cudaMalloc(&fftwArrDevice, N * N * N * sizeof(fftw_complex));

    int numElements = N * N * N;
    // Kernel dimension 1D, because 3D matrix stored in 1D array, just copying values at [i]
    int blockSize = 1024;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    cuToFftwComplexKernelGlobal<<<numBlocks, blockSize>>>(fftwArrDevice, cuArr, N);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA error at convertCuComplexToFFTW: " << cudaGetErrorString(err) << std::endl;
    }
    cudaMemcpy(fftwArr, fftwArrDevice, numElements * sizeof(fftw_complex), cudaMemcpyDeviceToHost);
    cudaFree(fftwArrDevice);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "[TIME][" << milliseconds << " ms]["<<N * N * N * sizeof(fftw_complex)<<"B] Converting (inkl. copy) cufftComplex to fftw_complex" << std::endl;
}




