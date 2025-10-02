#include "deconvolution/ThreadManager.h"




ThreadManager::ThreadManager(size_t maxNumberThreads, std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype, std::function<std::shared_ptr<IDeconvolutionBackend>()> createBackendFunction)
    : createBackendFunction_(createBackendFunction),
    algorithmPrototype_(std::move(algorithmPrototype)){
    threadpool = std::make_unique<ThreadPool>(maxNumberThreads);
    populate();
}

std::future<ComplexData> ThreadManager::registerTask(
    const ComplexData& psf,
    const ComplexData& image,
    std::function<ComplexData(const ComplexData&, const ComplexData&, const std::unique_ptr<DeconvolutionAlgorithm>&, std::shared_ptr<IDeconvolutionBackend>)> func){
        
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = algorithmPrototype_->clone();

        return threadpool->enqueue([this, func, psf, image, algorithm = std::move(algorithm)]() mutable {
            // Acquire backend when task actually runs
            std::shared_ptr<IDeconvolutionBackend> backend = getBackend();
            if (!backend) {
                throw std::runtime_error("No backend available");
            }
            
            try {
                ComplexData result = func(psf, image, algorithm, backend);
                returnBackend(backend);
                return result;
            } catch (...) {
                // Ensure backend is returned even on exception
                returnBackend(backend);
                throw;
            }
        });
    }

std::shared_ptr<IDeconvolutionBackend> ThreadManager::getBackend() {
    std::lock_guard<std::mutex> lock(backend_mutex);
    
    if (unusedBackends.empty()) {
        // Create new backend if none available (or return nullptr)
        return nullptr;
    }
    
    auto backend = unusedBackends.back();
    unusedBackends.pop_back();
    usedBackends.push_back(backend);
    
    return backend;
}

void ThreadManager::returnBackend(std::shared_ptr<IDeconvolutionBackend> backend) {
    std::lock_guard<std::mutex> lock(backend_mutex);
    
    auto it = std::find(usedBackends.begin(), usedBackends.end(), backend);
    if (it != usedBackends.end()) {
        usedBackends.erase(it);
        unusedBackends.push_back(backend);
    }
}

void ThreadManager::populate(){
    size_t memoryPerThread = algorithmPrototype_->getMemoryUsage();
    // TODO: Implement populate method logic
}