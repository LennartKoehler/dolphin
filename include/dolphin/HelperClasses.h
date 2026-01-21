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
#include <array>
#include <algorithm>

#include <sstream>
#include <iostream>
#include <cassert>
#include <map>
#include <memory>
#include <string>

#include "../lib/nlohmann/json.hpp"
#include "dolphinbackend/RectangleShape.h"
using json = nlohmann::json;struct Padding{
    RectangleShape before;
    RectangleShape after;
};




struct BoxCoord {
    RectangleShape position;
    RectangleShape dimensions;
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
        RectangleShape originalPosition = position;
        RectangleShape originalDimensions = dimensions;

        RectangleShape positionDiff = position - other.position;
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

        dimensions.updateVolume();
        
        // Calculate cropped amounts
        Padding croppedPadding;
        croppedPadding.before = position - originalPosition;
        
        croppedPadding.after.width = std::max(0, originalDimensions.width - dimensions.width - croppedPadding.before.width);
        croppedPadding.after.height = std::max(0, originalDimensions.height - dimensions.height - croppedPadding.before.height);
        croppedPadding.after.depth = std::max(0, originalDimensions.depth - dimensions.depth - croppedPadding.before.depth);
        
        assert(croppedPadding.before + croppedPadding.after + dimensions == originalDimensions && "CropTo something went wrong while cropping");

        return croppedPadding;
    }
};

struct BoxCoordWithPadding {
    BoxCoord box;
    Padding padding;
    bool isWithin(const BoxCoordWithPadding& other) const {
        return (this->getBox().isWithin(other.getBox()));
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


template<typename T>
struct Range {
    int start;
    int end;  // -1 means open-ended
    std::vector<T> values;
    
    // Default constructor
    Range() : start(0), end(0) {}
    
    Range(int s, int e, T value) : start(s), end(e) {
        values.push_back(std::forward<T>(value));
    }
    Range(int s, int e, std::vector<T> values) : start(s), end(e), values(values){}
    
    bool contains(int index) const {
        if (index == start && index == end){ return true; } // for the case of not a range, but just an index
        return index >= start && (end == -1 || index < end);
    }

    std::vector<T> get() const {
        return values;
    }
    
    // Add values from another range
    void addValues(const std::vector<T>& otherValues) {
        values.insert(values.end(), otherValues.begin(), otherValues.end());
    }
};
template<typename T>
class RangeMap {
private:
    std::vector<Range<T>> ranges;

public:
    RangeMap() = default;
    RangeMap(const RangeMap& other) : ranges(other.ranges) {}

    bool empty() { return ranges.empty(); }
    virtual void addRange(int start, int end, T value) {
        // Find if there's an existing range that matches exactly
        for (auto& range : ranges) {
            if (range.start == start && range.end == end) {
                range.values.push_back(std::forward<T>(value));
                return;
            }
        }
        
        // Create new range
        ranges.emplace_back(start, end, std::forward<T>(value));
    }

    std::vector<Range<T>> get(int index) const {
        std::vector<Range<T>> result;
        
        for (const auto& range : ranges) {
            if (range.contains(index)) {
                result.push_back(range);
            }
        }
        
        return result;
    }

    // Get pointers to the values without copying
    std::vector<const T*> getPointers(int index) const {
        std::vector<const T*> result;
        
        for (const auto& range : ranges) {
            if (range.contains(index)) {
                for (const auto& value : range.values) {
                    result.push_back(&value);
                }
            }
        }
        
        return result;
    }

    Range<T> operator[](int index) const {
        return get(index);
    }

    void clear() {
        ranges.clear();
    }
    
    void loadFromString(const std::string& config) {
        clear();
        
        // try {
        //     // First try to parse as JSON
        //     json jsonObj = json::parse(config);
        //     loadFromJSON(jsonObj);
        // } catch (const std::exception& e) {
            // If JSON parsing fails, try custom range format: "start:end[value1,value2]" or "start:[value]"
            try {
                parseCustomRangeFormat(config);
            } catch (const std::exception& e2) {
                // std::cerr << "[ERROR] Failed to parse string as JSON or custom format: " << e.what() << std::endl;
                throw std::invalid_argument("Invalid string format: " + config);
            }
        // }
    }

    // Load from JSON - unchanged
    void loadFromJSON(const json& jsonObj) {
        clear();
        
        for (auto it = jsonObj.begin(); it != jsonObj.end(); ++it) {
            const std::string& rangeKey = it.key();
            const json& valueArray = it.value();
            try {
                std::pair<int, int> rangePair = parseRange(rangeKey);
                int start = rangePair.first;
                int end = rangePair.second;
                
                if (valueArray.is_array()) {
                    for (const auto& valueJson : valueArray) {
                        if (valueJson.is_string()) {
                            T value = valueJson.get<std::string>();
                            addRange(start, end, value);
                        }
                    }
                } else if (valueArray.is_string()) {
                    T value = valueArray.get<std::string>();
                    addRange(start, end, value);
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to parse range '" << rangeKey 
                          << "': " << e.what() << std::endl;
            }
        }
    }

    // Iterator support - iterate over ranges, not individual indices
    class iterator {
    private:
        typename std::vector<Range<T>>::iterator rangeIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Range<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = Range<T>*;
        using reference = Range<T>&;
        
        iterator(typename std::vector<Range<T>>::iterator it) : rangeIt(it) {}
        
        reference operator*() { return *rangeIt; }
        pointer operator->() { return &(*rangeIt); }
        
        iterator& operator++() {
            ++rangeIt;
            return *this;
        }
        
        iterator operator++(int) {
            iterator temp = *this;
            ++rangeIt;
            return temp;
        }
        
        bool operator==(const iterator& other) const { return rangeIt == other.rangeIt; }
        bool operator!=(const iterator& other) const { return rangeIt != other.rangeIt; }
    };
    
    class const_iterator {
    private:
        typename std::vector<Range<T>>::const_iterator rangeIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const Range<T>;
        using difference_type = std::ptrdiff_t;
        using pointer = const Range<T>*;
        using reference = const Range<T>&;
        
        const_iterator(typename std::vector<Range<T>>::const_iterator it) : rangeIt(it) {}
        
        reference operator*() const { return *rangeIt; }
        pointer operator->() const { return &(*rangeIt); }
        
        const_iterator& operator++() {
            ++rangeIt;
            return *this;
        }
        
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++rangeIt;
            return temp;
        }
        
        bool operator==(const const_iterator& other) const { return rangeIt == other.rangeIt; }
        bool operator!=(const const_iterator& other) const { return rangeIt != other.rangeIt; }
    };

    // Iterator methods
    iterator begin() { return iterator(ranges.begin()); }
    iterator end() { return iterator(ranges.end()); }
    const_iterator begin() const { return const_iterator(ranges.begin()); }
    const_iterator end() const { return const_iterator(ranges.end()); }
    const_iterator cbegin() const { return const_iterator(ranges.begin()); }
    const_iterator cend() const { return const_iterator(ranges.end()); }

    // Find ranges that contain a specific index
    std::vector<iterator> findRangesContaining(int index) {
        std::vector<iterator> result;
        for (auto it = begin(); it != end(); ++it) {
            if (it->contains(index)) {
                result.push_back(it);
            }
        }
        return result;
    }

    // Debug method
    void debugPrint() const {
        std::cout << "[DEBUG] RangeMap contents:" << std::endl;
        for (const auto& range : ranges) {
            std::cout << "[DEBUG] Range " << range.start << ":";
            if (range.end == -1) {
                std::cout << "END";
            } else {
                std::cout << range.end;
            }
            std::cout << " -> ";
            for (const auto& value : range.values) {
                std::cout << "'" << value << "' ";
            }
            std::cout << std::endl;
        }
    }

private:
    // Fixed parseRange function
    std::pair<int, int> parseRange(const std::string& rangeStr) const {
        assert (!rangeStr.empty() && "Trying to parse empty string");
        size_t colonPos = rangeStr.find(':');
        int start, end;

        if (colonPos == std::string::npos) {
            start = std::stoi(rangeStr);
            end = std::stoi(rangeStr);
            return {start, end};
        }
        
        
        // Handle start
        if (colonPos == 0) {
            start = 0;
        } else {
            start = std::stoi(rangeStr.substr(0, colonPos));
        }
        
        // Handle end
        if (colonPos == rangeStr.length() - 1) {
            end = -1; // Open-ended
        } else {
            std::string endStr = rangeStr.substr(colonPos + 1);
            if (endStr.empty()) {
                end = -1;
            } else {
                end = std::stoi(endStr);
            }
        }
        
        return {start, end};
    }

    // Parse custom range format like "0:[constant_direct_0]" or "0:5[value1,value2]" 
    // Also supports multiple ranges like "0[constant_direct_0],1[constant_direct_1]"
    void parseCustomRangeFormat(const std::string& config) {
        // Split by commas first to handle multiple ranges
        std::vector<std::string> rangeBlocks;
        std::stringstream configStream(config);
        std::string block;
        
        // Parse blocks, being careful about brackets
        size_t pos = 0;
        while (pos < config.length()) {
            // Find the start of the next bracket
            size_t bracketStart = config.find('[', pos);
            if (bracketStart == std::string::npos) {
                break;
            }
            
            // Find the matching closing bracket
            size_t bracketEnd = config.find(']', bracketStart);
            if (bracketEnd == std::string::npos) {
                throw std::invalid_argument("Invalid custom range format - unmatched bracket: " + config);
            }
            
            // Extract this range block (including the bracket pair)
            std::string rangeBlock = config.substr(pos, bracketEnd - pos + 1);
            
            // Trim any leading comma and whitespace
            size_t blockStart = rangeBlock.find_first_not_of(", \t");
            if (blockStart != std::string::npos) {
                rangeBlock = rangeBlock.substr(blockStart);
            }
            
            if (!rangeBlock.empty()) {
                rangeBlocks.push_back(rangeBlock);
            }
            
            // Move position past this block
            pos = bracketEnd + 1;
            
            // Skip any comma and whitespace before the next block
            while (pos < config.length() && (config[pos] == ',' || config[pos] == ' ' || config[pos] == '\t')) {
                pos++;
            }
        }
        
        // Process each range block
        for (const auto& rangeBlock : rangeBlocks) {
            parseSingleRangeBlock(rangeBlock);
        }
    }

private:
    // Helper function to parse a single range block like "0[constant_direct_0]"
    void parseSingleRangeBlock(const std::string& rangeBlock) {
        // Find the bracket positions
        size_t bracketStart = rangeBlock.find('[');
        size_t bracketEnd = rangeBlock.find(']');
        
        if (bracketStart == std::string::npos || bracketEnd == std::string::npos || bracketEnd <= bracketStart) {
            throw std::invalid_argument("Invalid range block format: " + rangeBlock);
        }
        
        // Extract range part (before '[') and values part (inside '[...]')
        std::string rangeStr = rangeBlock.substr(0, bracketStart);
        std::string valuesStr = rangeBlock.substr(bracketStart + 1, bracketEnd - bracketStart - 1);
        
        // Parse the range
        auto [start, end] = parseRange(rangeStr);
        
        // Parse values (comma-separated within the brackets)
        std::stringstream ss(valuesStr);
        std::string value;
        
        while (std::getline(ss, value, ',')) {
            // Trim whitespace
            value.erase(0, value.find_first_not_of(" \t"));
            value.erase(value.find_last_not_of(" \t") + 1);
            
            if (!value.empty()) {
                addRange(start, end, static_cast<T>(value));
            }
        }
    }
};

// template <typename T>
// class NonoverlappingRangeMap : public RangeMap<T>{
//     void addRange(int start, int end, T value) override {
//         for (int i = start; i < end; i++) {
//             Range<T> range = getRange(i); // should actually only be one range as theire nonoverlapping
//             if ()
//         }        
//         // Create new range
//         ranges.emplace_back(start, end, std::forward<T>(value));
//     }

//     Range<T> getRange(int index){
//         Range<T> result;
        
//         for (const auto& range : ranges) {
//             if (range.contains(index)) {
//                 return range;
//             }
//         } 
//         return result;
       
//     }
// };