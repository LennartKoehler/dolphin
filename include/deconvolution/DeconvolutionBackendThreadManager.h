#pragma once

#include "IDeconvolutionBackend.h"
#include "algorithms/DeconvolutionAlgorithm.h"
#include "ThreadPool.h"

class DeconvolutionBackendThreadManager{
public:
    DeconvolutionBackendThreadManager(std::shared_ptr<ThreadPool> threadPool, size_t maxNumThreads, std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype, std::shared_ptr<IDeconvolutionBackend> backendPrototype);
    
    std::future<ComplexData> registerTask(
        std::vector<ComplexData>& psfs,
        ComplexData& image,
        std::function<ComplexData(std::vector<ComplexData>&, ComplexData&, std::unique_ptr<DeconvolutionAlgorithm>&, std::shared_ptr<IDeconvolutionBackend>)> func);

private:
    std::shared_ptr<IDeconvolutionBackend> getBackend();
    void returnBackend(std::shared_ptr<IDeconvolutionBackend> backend);
    void populate(size_t numberThreads);

    std::shared_ptr<IDeconvolutionBackend> backendPrototype_;
    std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype_;
    std::vector<std::shared_ptr<IDeconvolutionBackend>> unusedBackends;
    std::vector<std::shared_ptr<IDeconvolutionBackend>> usedBackends;
    std::mutex backend_mutex;
    std::shared_ptr<ThreadPool> threadpool;
    
};