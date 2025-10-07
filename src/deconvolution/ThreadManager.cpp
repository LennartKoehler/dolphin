#include "deconvolution/ThreadManager.h"



ThreadManager::ThreadManager(size_t maxNumberThreads, std::unique_ptr<DeconvolutionAlgorithm> algorithmPrototype, std::shared_ptr<IDeconvolutionBackend> backendPrototype)
    : backendPrototype_(backendPrototype),
    algorithmPrototype_(std::move(algorithmPrototype)){
    size_t numberThreads = getNumberThreads(maxNumberThreads);
    threadpool = std::make_unique<ThreadPool>(numberThreads);
    
    size_t threadPoolQueueSize = 300;
    ThreadPool* pool_ptr = threadpool.get();
    threadpool->setCondition([pool_ptr, threadPoolQueueSize]() -> bool {
        // std::cerr << pool_ptr->queueSize() << std::endl;
        return pool_ptr->queueSize() < threadPoolQueueSize;
     });
    populate(numberThreads);
}

std::future<ComplexData> ThreadManager::registerTask(
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

size_t ThreadManager::getNumberThreads(size_t maxNumberThreads){
    assert(backendPrototype_->isInitialized() && "[ERROR] ThreadManager: Backend must be initialized");

    size_t memoryPerCube = backendPrototype_->getWorkSize();
    size_t memoryMultiplier = algorithmPrototype_->getMemoryMultiplier();
    size_t memoryPerThread = memoryPerCube * (2 + memoryMultiplier); // * 2 for forward and backward fft plan;
    size_t availableMemory = backendPrototype_->getAvailableMemory();

    size_t numberThreads = availableMemory / memoryPerThread;
    return std::min(numberThreads, maxNumberThreads);

}

void ThreadManager::populate(size_t numberThreads){
    assert(backendPrototype_->isInitialized() && "[ERROR] ThreadManager: Backend must be initialized");
    for (int i = 0; i < numberThreads; i++){
       unusedBackends.emplace_back(backendPrototype_->clone());
    }

}

