#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include <functional>
#include <memory>
#include <mutex>
#include "backend/ComplexData.h"
#include "psf/PSF.h"
#include "backend/IBackend.h"
#include "Image3D.h"
#include <iostream>
class IBackend;

class PSFPreprocessor{
public:

    struct Key{
        RectangleShape shape;
        std::string psf;
    };

    struct KeyHash {
        std::size_t operator()(const Key& key) const {
            std::hash<std::string> hashFn;
            return hashFn(key.psf) ^ (std::hash<int>()(key.shape.width) << 1) ^ std::hash<int>()(key.shape.height);
        }
    };

    struct KeyEqual {
        bool operator()(const Key& lhs, const Key& rhs) const {
            return lhs.shape.width == rhs.shape.width &&
                   lhs.shape.height == rhs.shape.height &&
                   lhs.psf == rhs.psf;
        }
    };
    PSFPreprocessor() = default;

    void cleanup(){
        preprocessedPSFs.clear(); 
    };

    void setPreprocessingFunction(std::function<ComplexData*(RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend> backend)> func) {
        preprocessingFunction = std::move(func);
    }

    const ComplexData* getPreprocessedPSF(const RectangleShape& shape, const std::shared_ptr<PSF> psf, std::shared_ptr<IBackend> backend) {
        std::unique_lock<std::mutex> lock(mutex);
        Key key{shape, psf->ID};
        auto it = preprocessedPSFs.find(key);
        if (it != preprocessedPSFs.end()) {
            return it->second.get();
        }
        
        // PSF not found - create it
        std::shared_ptr<PSF> psfCopy = std::make_shared<PSF>(*psf);
        ComplexData* rawPtr = preprocessingFunction(shape, psfCopy, backend);
        
        // Insert into map
        psfBackends.push_back(backend); // hold on to backends until the psfs are also deleted
        auto [insertedIt, _] = preprocessedPSFs.emplace(
            Key{key},  // Copy the key
            std::unique_ptr<ComplexData>(rawPtr)
        );
        return rawPtr;
    }
    
private:
    std::mutex mutex;
    std::function<ComplexData*(const RectangleShape, std::shared_ptr<PSF>, std::shared_ptr<IBackend> backend)> preprocessingFunction;
    std::unordered_map<Key, std::unique_ptr<ComplexData>, KeyHash, KeyEqual> preprocessedPSFs;
    std::vector<std::shared_ptr<IBackend>> psfBackends;

};
namespace Preprocessor{


    std::vector<Image3D> splitImageHomogeneous(
        Image3D& image,
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape,
        const RectangleShape& imageShapePadded,
        const RectangleShape& cubeShapePadded);

    void expandToMinSize(Image3D& image, const RectangleShape& minSize);


    Padding padToShape(Image3D& image3D, const RectangleShape& targetShape, int borderType);
    void padImage(Image3D& image, const Padding& padding, int borderType);


}