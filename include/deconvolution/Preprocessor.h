#pragma once
#include <vector>
#include <opencv2/core/mat.hpp>
#include <functional>
#include <memory>

#include "backend/ComplexData.h"
#include "deconvolution/ImageMap.h"
#include "psf/PSF.h"

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

    void setPreprocessingFunction(std::function<ComplexData*(RectangleShape, std::shared_ptr<PSF>&)> func) {
        preprocessingFunction = std::move(func);
    }

    const ComplexData* getPreprocessedPSF(RectangleShape& shape, std::shared_ptr<PSF>& psf) {
        Key key{shape, psf->ID};

        auto it = preprocessedPSFs.find(key);
        if (it == preprocessedPSFs.end()) {
            ComplexData* rawPtr = preprocessingFunction(shape, psf); 
            // take ownership
            auto [insertedIt, _] = preprocessedPSFs.emplace(
                std::move(key), std::unique_ptr<ComplexData>(rawPtr)
            );
            return rawPtr;
        } else {
            return it->second.get();
        }
    }
private:
    std::function<ComplexData*(RectangleShape, std::shared_ptr<PSF>&)> preprocessingFunction;
    std::unordered_map<Key, std::unique_ptr<ComplexData>, KeyHash, KeyEqual> preprocessedPSFs;

};
namespace Preprocessor{
    std::vector<BoxCoord> splitImageHomogeneous(
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape);

    std::vector<std::vector<cv::Mat>> splitImageHomogeneous(
        std::vector<cv::Mat>& image,
        const RectangleShape& subimageShape,
        const RectangleShape& imageOriginalShape,
        const RectangleShape& imageShapePadded,
        const RectangleShape& cubeShapePadded);

    void expandToMinSize(std::vector<cv::Mat>& image, const RectangleShape& minSize);


    void padToShape(std::vector<cv::Mat>& image3D, const RectangleShape& targetShape, int borderType);


}