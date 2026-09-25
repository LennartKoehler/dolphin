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
#include <vector>

#include "dolphinbackend/CuboidShape.h"

struct Padding{
    CuboidShape before;
    CuboidShape after;
    bool operator==(const Padding& other) const {return (before == other.before && after == other.after);}

    size_t getPaddingWidthTotal() const { return before.width + after.width;}
    size_t getPaddingHeightTotal() const { return before.height + after.height;}
    size_t getPaddingDepthTotal() const { return before.depth + after.depth;}
    CuboidShape getTotalPadding() const {return before + after;}
};



struct PaddingScheme {
    Padding insidePadding;
    Padding imagePadding;
};

struct BoxCoord {
    CuboidPosition position;
    CuboidShape dimensions;
    bool operator==(const BoxCoord& other) const {return (position == other.position && dimensions == other.dimensions);}
    bool isWithin(const BoxCoord& other) const {
        for (size_t d = 0; d < 3; ++d) {
            if (position.at(d) < other.position.at(d) ||
                position.at(d) + static_cast<int64_t>(dimensions.at(d)) > other.position.at(d) + static_cast<int64_t>(other.dimensions.at(d)))
                return false;
        }
        return true;
    }
    Padding cropTo(const BoxCoord& other) {
        CuboidPosition originalPosition = position;
        CuboidShape originalDimensions = dimensions;

        Padding croppedPadding;
        for (size_t d = 0; d < 3; ++d) {
            int64_t positionDiff = std::min<int64_t>(0, position.at(d) - other.position.at(d));

            position.at(d) = std::max(position.at(d), other.position.at(d));

            int64_t maxDim = std::max<int64_t>(0, other.position.at(d) + static_cast<int64_t>(other.dimensions.at(d)) - position.at(d));

            dimensions.at(d) = static_cast<size_t>(std::min<int64_t>(static_cast<int64_t>(dimensions.at(d)) + positionDiff, maxDim));

            croppedPadding.before.at(d) = static_cast<size_t>(position.at(d) - originalPosition.at(d));
            croppedPadding.after.at(d) = static_cast<size_t>(std::max<int64_t>(0, static_cast<int64_t>(originalDimensions.at(d)) - static_cast<int64_t>(dimensions.at(d)) - static_cast<int64_t>(croppedPadding.before.at(d))));
        }

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
        return (this->getPaddedBox().isWithin(other.getPaddedBox()));
    }
    CuboidShape getPaddedShape() const {
        return box.dimensions + padding.getTotalPadding();
    }
    BoxCoord getPaddedBox() const {
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

inline std::vector<BoxCoord> subtractBox(const BoxCoord& outer, const BoxCoord& inner) {
    int64_t xStart = std::max<int64_t>(outer.position.width, inner.position.width);
    int64_t xEnd = std::min<int64_t>(outer.position.width + static_cast<int64_t>(outer.dimensions.width),
                                     inner.position.width + static_cast<int64_t>(inner.dimensions.width));
    int64_t yStart = std::max<int64_t>(outer.position.height, inner.position.height);
    int64_t yEnd = std::min<int64_t>(outer.position.height + static_cast<int64_t>(outer.dimensions.height),
                                     inner.position.height + static_cast<int64_t>(inner.dimensions.height));
    int64_t zStart = std::max<int64_t>(outer.position.depth, inner.position.depth);
    int64_t zEnd = std::min<int64_t>(outer.position.depth + static_cast<int64_t>(outer.dimensions.depth),
                                     inner.position.depth + static_cast<int64_t>(inner.dimensions.depth));

    if (xStart >= xEnd || yStart >= yEnd || zStart >= zEnd)
        return {outer};

    std::vector<BoxCoord> result;

    int64_t outerXEnd = outer.position.width + static_cast<int64_t>(outer.dimensions.width);
    int64_t outerYEnd = outer.position.height + static_cast<int64_t>(outer.dimensions.height);
    int64_t outerZEnd = outer.position.depth + static_cast<int64_t>(outer.dimensions.depth);

    if (xStart > outer.position.width)
        result.push_back({{outer.position.width, outer.position.height, outer.position.depth},
                          {static_cast<size_t>(xStart - outer.position.width), outer.dimensions.height, outer.dimensions.depth}});

    if (xEnd < outerXEnd)
        result.push_back({{xEnd, outer.position.height, outer.position.depth},
                          {static_cast<size_t>(outerXEnd - xEnd), outer.dimensions.height, outer.dimensions.depth}});

    if (yStart > outer.position.height)
        result.push_back({{xStart, outer.position.height, outer.position.depth},
                          {static_cast<size_t>(xEnd - xStart), static_cast<size_t>(yStart - outer.position.height), outer.dimensions.depth}});

    if (yEnd < outerYEnd)
        result.push_back({{xStart, yEnd, outer.position.depth},
                          {static_cast<size_t>(xEnd - xStart), static_cast<size_t>(outerYEnd - yEnd), outer.dimensions.depth}});

    if (zStart > outer.position.depth)
        result.push_back({{xStart, yStart, outer.position.depth},
                          {static_cast<size_t>(xEnd - xStart), static_cast<size_t>(yEnd - yStart), static_cast<size_t>(zStart - outer.position.depth)}});

    if (zEnd < outerZEnd)
        result.push_back({{xStart, yStart, zEnd},
                          {static_cast<size_t>(xEnd - xStart), static_cast<size_t>(yEnd - yStart), static_cast<size_t>(outerZEnd - zEnd)}});

    return result;
}
