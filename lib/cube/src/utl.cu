#include "utl.h"
#include "kernels.h"
#include <cuComplex.h>
#include <iostream>
#include <cstdlib>
#include <ctime>
#include <opencv2/opencv.hpp>

#ifdef ENABLE_CUBEUTL_DEBUG
#define DEBUG_LOG(msg) std::cout << msg << std::endl
#else
#define DEBUG_LOG(msg) // Nichts tun
#endif

namespace CUBE_UTL_INFO {
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
    void printSpecificElem(int index, fftw_complex* mat) {
        std::cout <<"[CHECK]["<< mat[index][0] << "+" << mat[index][1] <<"]" << std::endl;
    }
    void printRandomElem(int Nx, int Ny, int Nz, fftw_complex* mat) {
        int size = Nx * Ny * Nz;
        int randomIndex = rand() % size;
        std::cout <<"[CHECK]["<< mat[randomIndex][0] << "+" << mat[randomIndex][1] <<"]" << std::endl;
    }
}

namespace CUBE_UTL_CHECK {
    // Check Mat
    void checkUniformity(int Nx, int Ny, int Nz,fftw_complex* mat) {
        // Take the first element as a reference
        double *reference = mat[0];
        int countEqual = 0;
        int countDifferent = 0;
        double maxDeviation = 0.0;
        double totalDeviation = 0.0;
        int totalElements = Nx * Ny * Nz;

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
            std::cout << "[CHECK] Number of elements (" << Nx << "x" << Ny << "x" << Nz << "): "
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
    void displayHeatmap(int Nx, int Ny, int Nz, const fftw_complex* data) {
        int currentSlice = 0;  // Begin with the first slice

        // Create an OpenCV window (this remains open throughout the entire process)
        cv::namedWindow("Heatmap Slice Viewer", cv::WINDOW_NORMAL);

        while (true) {
            // Create a 2D array for the current slice
            cv::Mat heatmap(Ny, Nx, CV_32F);  // Correct heatmap size using Ny and Nx

            // Fill the heatmap with the values of the current slice
            for (int y = 0; y < Ny; ++y) {
                for (int x = 0; x < Nx; ++x) {
                    int idx = currentSlice * Ny * Nx + y * Nx + x;  // Correct indexing for 3D array
                    heatmap.at<float>(y, x) = std::abs(data[idx][0]);  // Using the real part as an example
                }
            }

            // Normalize the heatmap and convert it to 8-bit for display
            cv::normalize(heatmap, heatmap, 0, 255, cv::NORM_MINMAX);
            heatmap.convertTo(heatmap, CV_8U);

            // Create the title for the window (displays the slice) and show
            std::string title = "Slice " + std::to_string(currentSlice + 1) + " of " + std::to_string(Nz);
            cv::imshow("Heatmap Slice Viewer", heatmap);

            // Wait for a key press (waiting for 0ms means it waits for input immediately)
            int key = cv::waitKey(0);

            // If 'ENTER' (Code 13) is pressed, go to the next slice
            if (key == 13) {
                currentSlice = (currentSlice + 1) % Nz;  // Cycle through all slices
            }
            // If 'ESC' (Code 27) is pressed, break the loop and exit
            else if (key == 27) {
                break;
            }
        }
        cv::destroyWindow("Heatmap Slice Viewer");
    }
    bool checkOctantFourierShift(int Nx, int Ny, int Nz, fftw_complex* original, fftw_complex* shifted) {
        int width = Nx, height = Ny, depth = Nz;
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
    void printFftwComplexValueFromDevice(int idx, fftw_complex* fftwArr) {
        fftw_complex temp_host;
        cudaMemcpy(&temp_host, &fftwArr[idx], sizeof(fftw_complex), cudaMemcpyDeviceToHost);
        std::cout << "[CHECK] Element at index " << idx << ": "
                  << "Real: " << temp_host[0] << ", "
                  << "Imag: " << temp_host[1] << std::endl;
    }
    void printCufftComplexValueFromDevice(int idx, cufftComplex* cuArr) {
        cufftComplex temp_host;
        cudaMemcpy(&temp_host, &cuArr[idx], sizeof(cufftComplex), cudaMemcpyDeviceToHost);
        std::cout << "[CHECK] Element at index " << idx << ": "
                  << "Real: " << temp_host.x << ", "
                  << "Imag: " << temp_host.y << std::endl;
    }
}

namespace CUBE_UTL_INIT_MAT {
    // Mat initialization
    void createFftwUniformMat(int Nx, int Ny, int Nz, fftw_complex* mat){
#pragma omp parallel for
        for (int i = 0; i < Nx*Ny*Nz; i++) {
            mat[i][0] = 2.0f;  // real
            mat[i][1] = 0.0f;  // img
        }
    }
    void createFftwRandomMat(int Nx, int Ny, int Nz, fftw_complex* mat) {
        // Initialize the random number generator with the current time
        std::srand(static_cast<unsigned int>(std::time(0)));

        // Fill the matrix with random values in the range [0, 1] in steps of 0.001
#pragma omp parallel for
        for (int i = 0; i < Nx * Ny * Nz; i++) {
            // Generate a random number in the range [0, 1000] and divide by 1000 to get the range [0, 1]
            double randReal = (std::rand() % 1001) / 1000.0;
            double randImag = (std::rand() % 1001) / 1000.0;

            mat[i][0] = randReal;  // real
            mat[i][1] = randImag;  // img
        }
    }
    void createFftwSphereMat(int Nx, int Ny, int Nz, fftw_complex* mat) {
        // Determine the center of the matrix for each dimension
        int centerX = Nx / 2;
        int centerY = Ny / 2;
        int centerZ = Nz / 2;

        // Find the smallest dimension to ensure the sphere fits
        int radius = std::min({Nx, Ny, Nz}) / 2;

#pragma omp parallel for
        for (int i = 0; i < Nx * Ny * Nz; i++) {
            // Calculate 3D coordinates (x, y, z) from the 1D index
            int z = i / (Nx * Ny);
            int y = (i % (Nx * Ny)) / Nx;
            int x = i % Nx;

            // Calculate the squared distance from the center
            int distSq = (x - centerX) * (x - centerX) +
                         (y - centerY) * (y - centerY) +
                         (z - centerZ) * (z - centerZ);

            // If the distance is within the radius, set the value to 1, otherwise set it to 0
            if (distSq <= radius * radius) {
                mat[i][0] = 1.0f;  // Real part
                mat[i][1] = 0.0f;  // Imaginary part
            } else {
                mat[i][0] = 0.0f;  // Real part
                mat[i][1] = 0.0f;  // Imaginary part
            }
        }
    }
}

namespace CUBE_UTL_COPY {
    // Copying fftw_complex datatype to GPU
    void copyDataFromHostToDevice(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMemcpy(dest, src, sizeof(fftw_complex)*Nx*Ny*Nz, cudaMemcpyHostToDevice);

        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(fftw_complex)*Nx*Ny*Nz<<"B] Copy Data from Host to Device");

    }
    void copyDataFromDeviceToHost(int Nx, int Ny, int Nz,fftw_complex* dest, fftw_complex* src) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        cudaMemcpy(dest, src, sizeof(fftw_complex)*Nx*Ny*Nz, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
        }

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<sizeof(fftw_complex)*Nx*Ny*Nz<<"B] Copy Data from Device to Host");
    }
}

namespace CUBE_UTL_CONVERT {
    // Conversions
    void convertFftwToCuComplexOnDevice(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fftw_complex* fftwArrDevice;
        cudaMalloc(&fftwArrDevice, Nx * Ny * Nz * sizeof(fftw_complex));

        CUBE_UTL_COPY::copyDataFromHostToDevice(Nx, Ny, Nz, fftwArrDevice, fftwArr);

        dim3 blockSize(16, 8, 8);  // Blockgröße (optimal anpassen)
        dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                      (Ny + blockSize.y - 1) / blockSize.y,
                      (Nz + blockSize.z - 1) / blockSize.z);  // Grid-Größe

        fftwToCuComplexKernelGlobal<<<gridSize, blockSize>>>(Nx, Ny, Nz, cuArr, fftwArrDevice);
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
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<Nx * Ny * Nz * sizeof(fftw_complex)<<"B] Converting (inkl. copy) fftw_complex to cuComplex");
    }
    void convertFftwToCufftComplexOnDevice(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cufftComplex* cuArr) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fftw_complex* fftwArrDevice;
        cudaMalloc(&fftwArrDevice, Nx * Ny * Nz * sizeof(fftw_complex));
        CUBE_UTL_COPY::copyDataFromHostToDevice(Nx, Ny, Nz, fftwArrDevice, fftwArr);

        dim3 blockSize(16, 8, 8);  // Blockgröße (optimal anpassen)
        dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                      (Ny + blockSize.y - 1) / blockSize.y,
                      (Nz + blockSize.z - 1) / blockSize.z);  // Grid-Größe

        fftwToCufftComplexKernelGlobal<<<gridSize, blockSize>>>(Nx, Ny, Nz, cuArr, fftwArrDevice);

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
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<Nx * Ny * Nz * sizeof(fftw_complex)<<"B] Converting (inkl. copy) fftw_complex to cufftComplex");
    }
    void convertCuToFftwComplexOnHost(int Nx, int Ny, int Nz,fftw_complex* fftwArr, cuComplex* cuArr) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fftw_complex* fftwArrDevice;
        cudaMalloc(&fftwArrDevice, Nx * Ny * Nz * sizeof(fftw_complex));

        dim3 blockSize(16, 8, 8);  // Blockgröße (optimal anpassen)
        dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                      (Ny + blockSize.y - 1) / blockSize.y,
                      (Nz + blockSize.z - 1) / blockSize.z);  // Grid-Größe

        cuToFftwComplexKernelGlobal<<<gridSize, blockSize>>>(Nx, Ny, Nz, fftwArrDevice, cuArr);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at convertCuComplexToFFTW: " << cudaGetErrorString(err) << std::endl;
        }
        cudaMemcpy(fftwArr, fftwArrDevice, Nx*Ny*Nz * sizeof(fftw_complex), cudaMemcpyDeviceToHost);
        cudaFree(fftwArrDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<Nx * Ny* Nz * sizeof(fftw_complex)<<"B] Converting (inkl. copy) cuComplex to fftw_complex");
    }
    void convertCufftToFftwComplexOnHost(int Nx, int Ny, int Nz, fftw_complex* fftwArr, cufftComplex* cuArr) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        fftw_complex* fftwArrDevice;
        cudaMalloc(&fftwArrDevice, Nx * Ny * Nz * sizeof(fftw_complex));

        dim3 blockSize(16, 8, 8);  // Blockgröße (optimal anpassen)
        dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x,
                      (Ny + blockSize.y - 1) / blockSize.y,
                      (Nz + blockSize.z - 1) / blockSize.z);  // Grid-Größe

        cufftToFftwComplexKernelGlobal<<<gridSize, blockSize>>>(Nx, Ny, Nz, fftwArrDevice, cuArr);
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            std::cerr << "CUDA error at convertCuComplexToFFTW: " << cudaGetErrorString(err) << std::endl;
        }
        CUBE_UTL_COPY::copyDataFromDeviceToHost(Nx, Ny, Nz, fftwArr, fftwArrDevice);
        cudaFree(fftwArrDevice);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        DEBUG_LOG("[TIME][" << milliseconds << " ms]["<<Nx * Ny * Nz * sizeof(fftw_complex)<<"B] Converting (inkl. copy) cufftComplex to fftw_complex");
    }
}




