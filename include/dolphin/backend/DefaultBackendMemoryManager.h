#pragma once
#include "dolphinbackend/IBackendMemoryManager.h"


class DefaultBackendMemoryManager : public IBackendMemoryManager{
public:
    // Constructor
    DefaultBackendMemoryManager();
    ~DefaultBackendMemoryManager();
    

    std::string getDeviceType() const noexcept override {
        return "default";
    }
    
    void sync() override {}
    // Memory management initialization
    
    // Data management
    void memCopy(const ComplexData& srcdata, ComplexData& destdata) const override;
    void allocateMemoryOnDevice(ComplexData& data) const override;
    ComplexData allocateMemoryOnDevice(const RectangleShape& shape) const override;
    bool isOnDevice(void* data) const override;
    ComplexData copyData(const ComplexData& srcdata) const override;
    ComplexData copyDataToDevice(const ComplexData& srcdata) const override;
    ComplexData moveDataFromDevice(const ComplexData& srcdata, const IBackendMemoryManager& destBackend) const override;
    void freeMemoryOnDevice(ComplexData& data) const override;

private:

};