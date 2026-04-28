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
#include <vector>
#include <functional>
#include <memory>
#include <mutex>
#include <iostream>

#include "dolphin/deconvolution/DeconvolutionConfig.h"
#include "dolphinbackend/ComplexData.h"
#include "dolphin/psf/PSF.h"
#include "dolphinbackend/IBackend.h"
#include "dolphin/Image3D.h"
#include "dolphin/backend/BackendFactory.h"

class IBackend;

class PSFPreprocessor{
public:

    struct Key{
        CuboidShape shape;
        std::string psf;
        std::string device;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const {
            std::hash<std::string> hashFn;
            return hashFn(key.psf) ^ hashFn(key.device) ^ (std::hash<int>()(key.shape.width) << 1) ^ std::hash<int>()(key.shape.height);
        }
    };

    struct KeyEqual {
        bool operator()(const Key& lhs, const Key& rhs) const {
            return lhs.shape.width == rhs.shape.width &&
                   lhs.shape.height == rhs.shape.height &&
                   lhs.device == rhs.device &&
                   lhs.psf == rhs.psf;
        }
    };
    PSFPreprocessor() = default;

    ~PSFPreprocessor(){
        cleanup();
    }
    void cleanup(){
        for (auto& [key, psf] : preprocessedPSFs){
            psf.reset();
        }
    };

    void setPreprocessingFunction(std::function<std::unique_ptr<ComplexData>(CuboidShape, std::shared_ptr<PSF>, IBackend& backend)> func) {
        preprocessingFunction = std::move(func);
    }

    const ComplexData* getPreprocessedPSF(const CuboidShape& shape, const std::shared_ptr<PSF> psf, IBackend& backend) {
        std::unique_lock<std::mutex> lock(mutex);
        Key key{shape, psf->ID, backend.getDeviceString()};
        auto it = preprocessedPSFs.find(key);
        if (it != preprocessedPSFs.end()) {
            return it->second.get();
        }

        // PSF not found - create it
        std::shared_ptr<PSF> psfCopy = std::make_shared<PSF>(*psf);
        std::unique_ptr<ComplexData> preprocessedPSF = preprocessingFunction(shape, psfCopy, backend);

        // Insert into map
        // psfBackends.push_back(backend); // hold on to backends until the psfs are also deleted
        auto [insertedIt, _] = preprocessedPSFs.emplace(
            Key{key},  // Copy the key
            std::move(preprocessedPSF)
        );
        it = preprocessedPSFs.find(key);
        if (it != preprocessedPSFs.end()) {
            return it->second.get();
        }
        return nullptr;
    }

private:
    std::mutex mutex;
    std::function<std::unique_ptr<ComplexData>(const CuboidShape, std::shared_ptr<PSF>, IBackend& backend)> preprocessingFunction;
    std::unordered_map<Key, std::unique_ptr<ComplexData>, KeyHash, KeyEqual> preprocessedPSFs;
    // std::vector<IBackend&> psfBackends;

};
namespace Preprocessor{

    ComplexData convertImageToComplexData(const Image3D& image);
    Image3D convertComplexDataToImage(const ComplexData& data);

    RealData convertImageToRealData(const Image3D& image, IBackendMemoryManager& memoryManager);
    Image3D convertRealDataToImage(const RealData& data);
    void expandToMinSize(Image3D& image, const CuboidShape& minSize);


    Padding padToShape(Image3D& image3D, const CuboidShape& targetShape, PaddingFillType borderType);

    void padImage(Image3D& image, const Padding& padding, PaddingFillType borderType, float shapeScale = 1.0f);
    void padImageMirror(Image3D& image, const Padding& padding);
    void padImageZero(Image3D& image, const Padding& padding);
    void padImageLinear(Image3D& image, const Padding& padding);
    void padImageQuadratic(Image3D& image, const Padding& padding, float shapeScale = 1.0f);
    void padImageSinusoid(Image3D& image, const Padding& padding);
    void padImageGaussian(Image3D& image, const Padding& padding, float shapeScale = 1.0f);


}
