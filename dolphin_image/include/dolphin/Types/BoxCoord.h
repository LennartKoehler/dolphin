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

    int getPaddingWidthTotal() const { return before.width + after.width;}
    int getPaddingHeightTotal() const { return before.height + after.height;}
    int getPaddingDepthTotal() const { return before.width + after.width;}
    CuboidShape getTotalPadding() const {return before + after;}
};



struct BoxCoord {
    CuboidShape position;
    CuboidShape dimensions;
    bool isWithin(const BoxCoord& other) const {
        // Check if this box is completely within the other box
        return (position.width >= other.position.width &&
                position.height >= other.position.height &&
                position.depth >= other.position.depth &&
                position.width + dimensions.width <= other.position.width + other.dimensions.width &&
                position.height + dimensions.height <= other.position.height + other.dimensions.height &&
                position.depth + dimensions.depth <= other.position.depth + other.dimensions.depth);
    }
    // TODO i think there is a better way to do this
    Padding cropTo(const BoxCoord& other) {
        // Store original values to calculate what was cropped
        CuboidShape originalPosition = position;
        CuboidShape originalDimensions = dimensions;

        CuboidShape positionDiff = position - other.position;
        positionDiff.clamp({0,0,0});
        // Adjust position to be within the other box
        position.width = std::max(position.width, other.position.width);
        position.height = std::max(position.height, other.position.height);
        position.depth = std::max(position.depth, other.position.depth);


        // Calculate the maximum allowed dimensions
        int maxWidth = std::max(0, other.position.width + other.dimensions.width - position.width);
        int maxHeight = std::max(0, other.position.height + other.dimensions.height - position.height);
        int maxDepth = std::max(0, other.position.depth + other.dimensions.depth - position.depth);

        // Crop dimensions to fit within the other box
        // have to accomodate for something being cut off at the beginning aswell, which therefore reduces the desired dimensions
        dimensions.width = std::min(dimensions.width + positionDiff.width, maxWidth);
        dimensions.height = std::min(dimensions.height + positionDiff.height, maxHeight);
        dimensions.depth = std::min(dimensions.depth + positionDiff.depth, maxDepth);


        // Calculate cropped amounts
        Padding croppedPadding;
        croppedPadding.before = position - originalPosition;

        croppedPadding.after.width = std::max(0, originalDimensions.width - dimensions.width - croppedPadding.before.width);
        croppedPadding.after.height = std::max(0, originalDimensions.height - dimensions.height - croppedPadding.before.height);
        croppedPadding.after.depth = std::max(0, originalDimensions.depth - dimensions.depth - croppedPadding.before.depth);

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
