#pragma once
#include "ThreadPool.h"
#include "IDeconvolutionBackend.h"
#include "algorithms/DeconvolutionAlgorithm.h"


class ThreadManager{
public:
    ThreadManager(size_t maxNumThreads, std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype, std::shared_ptr<IDeconvolutionBackend> backendPrototype);
    
    std::future<ComplexData> registerTask(
        const ComplexData& psf,
        const ComplexData& image,
        std::function<ComplexData(const ComplexData&, const ComplexData&, const std::unique_ptr<DeconvolutionAlgorithm>&, std::shared_ptr<IDeconvolutionBackend>)> func);

private:
    size_t getAvailableMemory();
    std::shared_ptr<IDeconvolutionBackend> getBackend();
    void returnBackend(std::shared_ptr<IDeconvolutionBackend> backend);
    void populate();
    
    std::shared_ptr<IDeconvolutionBackend> backendPrototype_;
    std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype_;
    std::vector<std::shared_ptr<IDeconvolutionBackend>> unusedBackends;
    std::vector<std::shared_ptr<IDeconvolutionBackend>> usedBackends;
    std::mutex backend_mutex;
    std::unique_ptr<ThreadPool> threadpool;
    
};