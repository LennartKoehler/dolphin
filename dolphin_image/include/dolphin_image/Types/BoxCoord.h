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

#include <algorithm>
#include <cassert>
#include <string>

#include "dolphinbackend/CuboidShape.h"

struct Padding{
    CuboidShape before;
    CuboidShape after;

    size_t getPaddingWidthTotal() const { return before.width + after.width;}
    size_t getPaddingHeightTotal() const { return before.height + after.height;}
    size_t getPaddingDepthTotal() const { return before.depth + after.depth;}
    CuboidShape getTotalPadding() const {return before + after;}
};



struct BoxCoord {
    CuboidPosition position;
    CuboidShape dimensions;
    bool isWithin(const BoxCoord& other) const {
        return (position.width >= other.position.width &&
                position.height >= other.position.height &&
                position.depth >= other.position.depth &&
                position.width + static_cast<int64_t>(dimensions.width) <= other.position.width + static_cast<int64_t>(other.dimensions.width) &&
                position.height + static_cast<int64_t>(dimensions.height) <= other.position.height + static_cast<int64_t>(other.dimensions.height) &&
                position.depth + static_cast<int64_t>(dimensions.depth) <= other.position.depth + static_cast<int64_t>(other.dimensions.depth));
    }
    Padding cropTo(const BoxCoord& other) {
        CuboidPosition originalPosition = position;
        CuboidShape originalDimensions = dimensions;

        CuboidPosition positionDiff = position - other.position;
        positionDiff.width = std::min<int64_t>(0, positionDiff.width);
        positionDiff.height = std::min<int64_t>(0, positionDiff.height);
        positionDiff.depth = std::min<int64_t>(0, positionDiff.depth);

        position.width = std::max(position.width, other.position.width);
        position.height = std::max(position.height, other.position.height);
        position.depth = std::max(position.depth, other.position.depth);

        int64_t maxWidth = std::max<int64_t>(0, other.position.width + static_cast<int64_t>(other.dimensions.width) - position.width);
        int64_t maxHeight = std::max<int64_t>(0, other.position.height + static_cast<int64_t>(other.dimensions.height) - position.height);
        int64_t maxDepth = std::max<int64_t>(0, other.position.depth + static_cast<int64_t>(other.dimensions.depth) - position.depth);

        dimensions.width = static_cast<size_t>(std::min<int64_t>(static_cast<int64_t>(dimensions.width) + positionDiff.width, maxWidth));
        dimensions.height = static_cast<size_t>(std::min<int64_t>(static_cast<int64_t>(dimensions.height) + positionDiff.height, maxHeight));
        dimensions.depth = static_cast<size_t>(std::min<int64_t>(static_cast<int64_t>(dimensions.depth) + positionDiff.depth, maxDepth));

        Padding croppedPadding;
        croppedPadding.before = (position - originalPosition).toShape();

        croppedPadding.after.width = static_cast<size_t>(std::max<int64_t>(0, static_cast<int64_t>(originalDimensions.width) - static_cast<int64_t>(dimensions.width) - static_cast<int64_t>(croppedPadding.before.width)));
        croppedPadding.after.height = static_cast<size_t>(std::max<int64_t>(0, static_cast<int64_t>(originalDimensions.height) - static_cast<int64_t>(dimensions.height) - static_cast<int64_t>(croppedPadding.before.height)));
        croppedPadding.after.depth = static_cast<size_t>(std::max<int64_t>(0, static_cast<int64_t>(originalDimensions.depth) - static_cast<int64_t>(dimensions.depth) - static_cast<int64_t>(croppedPadding.before.depth)));

        assert(croppedPadding.before + croppedPadding.after + dimensions == originalDimensions && "CropTo something went wrong while cropping");

        return croppedPadding;
    }
    std::string print() const {
        return "Position: " + position.print() + "; Dimensions: " + dimensions.print();
    }
};

struct BoxCoordWithPadding {
    BoxCoord box;
    Padding padding;
    bool isWithin(const BoxCoordWithPadding& other) const {
        return (this->getBox().isWithin(other.getBox()));
    }
    CuboidShape getPaddedShape() const {
        return box.dimensions + padding.getTotalPadding();
    }
    BoxCoord getBox() const {
        return BoxCoord{this->box.position - this->padding.before, this->box.dimensions + this->padding.before + this->padding.after};
    }
};

template<typename entryType>
struct BoxEntryPair {
    BoxCoord box;
    entryType entry;

    BoxEntryPair(BoxCoord b, entryType p)
        : box(b), entry(std::move(p)) {}
};
