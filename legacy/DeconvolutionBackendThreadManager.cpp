#include "deconvolution/DeconvolutionBackendThreadManager.h"



DeconvolutionBackendThreadManager::DeconvolutionBackendThreadManager(std::shared_ptr<ThreadPool> threadPool, size_t numberThreads, std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype, std::shared_ptr<IDeconvolutionBackend> backendPrototype)
    : threadpool(threadPool),
    backendPrototype_(backendPrototype),
    algorithmPrototype_(std::move(algorithmPrototype)){
    
    size_t threadPoolQueueSize = numberThreads + 10;
    ThreadPool* pool_ptr = threadpool.get();

    threadpool->setCondition([pool_ptr, threadPoolQueueSize]() -> bool {
        return pool_ptr->queueSize() < threadPoolQueueSize;
     });
    populate(numberThreads);
}

std::future<ComplexData> DeconvolutionBackendThreadManager::registerTask(
    std::vector<ComplexData>& psfs,
    ComplexData& image,
    std::function<ComplexData(std::vector<ComplexData>&, ComplexData&, std::unique_ptr<DeconvolutionAlgorithm>&, std::shared_ptr<IDeconvolutionBackend>)> func){
        
        std::unique_ptr<DeconvolutionAlgorithm> algorithm = algorithmPrototype_->clone();

        return threadpool->enqueue([this, func, psfs, image, algorithm = std::move(algorithm)]() mutable {
            // Acquire backend when task actually runs
            std::shared_ptr<IDeconvolutionBackend> backend = getBackend();
            algorithm->setBackend(backend);

            if (!backend) {
                throw std::runtime_error("No backend available");
            }
            
            try {
                ComplexData result = func(psfs, image, algorithm, backend);
                returnBackend(backend);
                return result;
            } catch (...) {
                // Ensure backend is returned even on exception
                returnBackend(backend);
                throw;
            }
        });
    }

std::shared_ptr<IDeconvolutionBackend> DeconvolutionBackendThreadManager::getBackend() {
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

void DeconvolutionBackendThreadManager::returnBackend(std::shared_ptr<IDeconvolutionBackend> backend) {
    std::lock_guard<std::mutex> lock(backend_mutex);
    
    auto it = std::find(usedBackends.begin(), usedBackends.end(), backend);
    if (it != usedBackends.end()) {
        usedBackends.erase(it);
        unusedBackends.push_back(backend);
    }
}



void DeconvolutionBackendThreadManager::populate(size_t numberThreads){
    assert(backendPrototype_->isInitialized() && "[ERROR] DeconvolutionBackendThreadManager: Backend must be initialized");
    for (int i = 0; i < numberThreads; i++){
       unusedBackends.emplace_back(backendPrototype_->clone());
    }

}

