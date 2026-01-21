/*
Copyright by Lennart Koehler

Research Group Applied Systems Biology - Head: Prof. Dr. Marc Thilo Figge
https://www.leibniz-hki.de/en/applied-systems-biology.html
HKI-Center for Systems Biology of Infection
Leibniz Institute for Natural Product Research and Infection Biology - Hans Knöll Institute (HKI)
Adolf-Reichwein-Straße 23, 07745 Jena, Germany

The project code is licensed under the MIT license.
See the LICENSE file provided with the code for the full license.
*/

#pragma once
#include "dolphinbackend/IBackendMemoryManager.h"


class DefaultBackendMemoryManager : public IBackendMemoryManager{
public:

    static DefaultBackendMemoryManager& getInstance(); 

    std::string getDeviceString() const noexcept override {
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

    // Constructor
    DefaultBackendMemoryManager();
    ~DefaultBackendMemoryManager();
};