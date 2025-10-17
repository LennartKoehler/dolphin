#pragma once

#include <string>
#include <opencv2/core/base.hpp>
#include <vector>
#include "Config.h"
#include "DeconvolutionAlgorithmFactory.h"
#include "../backend/BackendFactory.h"
#include <map>
#include <memory>


template<typename T>
class RangeMap {
public:
    struct Range {
        int start;
        int end;  // -1 means open-ended
        std::vector<T> values;
        
        Range(int s, int e, T value) : start(s), end(e) {
            values.push_back(std::forward<T>(value));
        }
        
        bool contains(int index) const {
            return index >= start && (end == -1 || index < end);
        }

        std::vector<T> get() const {
            return values;
        }
    };

private:
    std::vector<Range> ranges;

public:
    RangeMap() = default;
    RangeMap(const RangeMap& other) : ranges(other.ranges) {}

    bool empty() { return ranges.empty(); }
    void addRange(int start, int end, T value) {
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

    std::vector<T> get(int index) const {
        std::vector<T> result;
        
        for (const auto& range : ranges) {
            if (range.contains(index)) {
                result.insert(result.end(), range.values.begin(), range.values.end());
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

    std::vector<T> operator[](int index) const {
        return get(index);
    }

    void clear() {
        ranges.clear();
    }

    // Load from JSON - unchanged
    void loadFromJSON(const json& jsonObj) {
        clear();
        
        for (auto& [rangeKey, valueArray] : jsonObj.items()) {
            try {
                auto [start, end] = parseRange(rangeKey);
                
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
        typename std::vector<Range>::iterator rangeIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = Range;
        using difference_type = std::ptrdiff_t;
        using pointer = Range*;
        using reference = Range&;
        
        iterator(typename std::vector<Range>::iterator it) : rangeIt(it) {}
        
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
        typename std::vector<Range>::const_iterator rangeIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const Range;
        using difference_type = std::ptrdiff_t;
        using pointer = const Range*;
        using reference = const Range&;
        
        const_iterator(typename std::vector<Range>::const_iterator it) : rangeIt(it) {}
        
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
        size_t colonPos = rangeStr.find(':');
        if (colonPos == std::string::npos) {
            throw std::invalid_argument("Invalid range format: " + rangeStr);
        }
        
        int start, end;
        
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
};




class DeconvolutionConfig : public Config{
public:
    DeconvolutionConfig();
    DeconvolutionConfig(const DeconvolutionConfig& other);

    // Use the struct for parameters
    std::string algorithmName = "RichardsonLucyTotalVariation";
    int subimageSize = 0;
    int iterations = 10;
    float epsilon = 1e-6;
    bool grid = false;
    float lambda = 0.001;
    int borderType = cv::BORDER_REFLECT;
    bool time = false;
    bool saveSubimages = false;
    std::string backenddeconv = "cpu";
    int nThreads = 1;
    float maxMemGB = 0;
    bool verbose = false;

    RangeMap<std::string> cubePSFMap;
    RangeMap<std::string> layerPSFMap;




    virtual bool loadFromJSON(const json& jsonData) override;
    virtual json writeToJSON() override;

private:
    void registerAllParameters();
};



