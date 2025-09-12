#pragma once
#include "IDeconvolutionBackend.h"

class GPUBackend : public IDeconvolutionBackend{
    void preprocess() override;
    void postprocess() override;
};