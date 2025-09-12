#include "deconvolution/backend/GPUBackend.h"
#include <omp.h>

void GPUBackend::preprocess(){
    movePSFstoGPU();
    omp_set_num_threads(1);
}

std::unordered_map<PSFIndex, PSFfftw*>& GPUBackend::movePSFstoGPU(std::unordered_map<PSFIndex, PSFfftw*>& psfMap){
    try{     
        for (auto it : psfMap){
            fftw_complex *d_temp_h;
             // psf same size as cube
            cudaMalloc((void**)&d_temp_h, cubeMetaData.cubeVolume * sizeof(PSFfftw));
            CUBE_UTL_COPY::copyDataFromHostToDevice(cubeMetaData.cubeWidth, cubeMetaData.cubeHeight, cubeMetaData.cubeDepth, d_temp_h, it.second);

            it.second = d_temp_h; 
        } 
        
    } catch (const exception& e) {
        cerr << "[ERROR] Exception in GPU preprocessing: " << e.what() << endl;
     }

}