#pragma once

#include <string>
#include <opencv2/core/base.hpp>
#include <vector>
#include "Config.h"

#include <map>
#include <memory>

template<typename T>
class RangeMap {    
public:
    RangeMap() = default;
    RangeMap(const RangeMap& other)
        : ranges(other.ranges){}

    void addRange(int start, int end, T value) {
        for (int i = start; i < end; ++i) {
            ranges[i].push_back(value);
        }
    }
    
    std::vector<T>& get(int index){
        return ranges[index];
    }
    
    std::vector<T>& operator[](int index) {
        return get(index);
    }

    
    
    void clear() {
        ranges.clear();
    }
    
    // Load from JSON - this is the key function
    void loadFromJSON(const json& jsonObj) {
        clear();
        
        
        // Iterate through all key-value pairs in the JSON object
        for (auto& [rangeKey, valueArray] : jsonObj.items()) {
            try {
                // Parse the range key "start:end"
                auto [start, end] = parseRange(rangeKey);
                
                // Parse the value array
                if (valueArray.is_array()) {
                    for (const auto& valueJson : valueArray) {
                        if (valueJson.is_string()) {
                            T value = valueJson.get<std::string>();
                            addRange(start, end, value);
                        }
                    }
                } else if (valueArray.is_string()) {
                    // Handle single string value
                    T value = valueArray.get<std::string>();
                    addRange(start, end, value);
                }
            } catch (const std::exception& e) {
                std::cerr << "[ERROR] Failed to parse range '" << rangeKey 
                          << "': " << e.what() << std::endl;
            }
        }
    }
    


private:
    std::map<int, std::vector<T>> ranges;
    
    // Helper function to parse range string "start:end" 
    std::pair<int, int> parseRange(const std::string& rangeStr) const {
        size_t colonPos = rangeStr.find(':');
        if (colonPos == std::string::npos) {
            throw std::invalid_argument("Invalid range format: " + rangeStr);
        }
        
        int start = std::stoi(rangeStr.substr(0, colonPos));
        int end = std::stoi(rangeStr.substr(colonPos + 1));
        
        return {start, end};
    }
public:
    class iterator {
    private:
        typename std::map<int, std::vector<T>>::iterator mapIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = std::pair<const int, std::vector<T>>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        
        iterator(typename std::map<int, std::vector<T>>::iterator it) : mapIt(it) {}
        
        reference operator*() { return *mapIt; }
        pointer operator->() { return &(*mapIt); }
        
        iterator& operator++() {
            ++mapIt;
            return *this;
        }
        
        iterator operator++(int) {
            iterator temp = *this;
            ++mapIt;
            return temp;
        }
        
        bool operator==(const iterator& other) const { return mapIt == other.mapIt; }
        bool operator!=(const iterator& other) const { return mapIt != other.mapIt; }
    };
    
    class const_iterator {
    private:
        typename std::map<int, std::vector<T>>::const_iterator mapIt;
        
    public:
        using iterator_category = std::forward_iterator_tag;
        using value_type = const std::pair<const int, std::vector<T>>;
        using difference_type = std::ptrdiff_t;
        using pointer = value_type*;
        using reference = value_type&;
        
        const_iterator(typename std::map<int, std::vector<T>>::const_iterator it) : mapIt(it) {}
        
        reference operator*() const { return *mapIt; }
        pointer operator->() const { return &(*mapIt); }
        
        const_iterator& operator++() {
            ++mapIt;
            return *this;
        }
        
        const_iterator operator++(int) {
            const_iterator temp = *this;
            ++mapIt;
            return temp;
        }
        
        bool operator==(const const_iterator& other) const { return mapIt == other.mapIt; }
        bool operator!=(const const_iterator& other) const { return mapIt != other.mapIt; }
    };

    iterator find(int index) { return iterator(ranges.find(index)); }
    const_iterator find(int index) const { return const_iterator(ranges.find(index)); }
    iterator begin() { return iterator(ranges.begin()); }
    iterator end() { return iterator(ranges.end()); }
    const_iterator begin() const { return const_iterator(ranges.begin()); }
    const_iterator end() const { return const_iterator(ranges.end()); }
    const_iterator cbegin() const { return const_iterator(ranges.begin()); }
    const_iterator cend() const { return const_iterator(ranges.end()); }
};


class DeconvolutionConfig : public Config{
public:
    DeconvolutionConfig();
    DeconvolutionConfig(const DeconvolutionConfig& other);
    std::string algorithmName = "RichardsonLucyTotalVariation";
    int subimageSize = 0; //sub-image size (edge)
    int iterations = 10;
    double epsilon = 1e-6;
    bool grid = false;
    double lambda = 0.001;
    int borderType = cv::BORDER_REFLECT;
    int psfSafetyBorder = 10;
    int cubeSize = 0;
    // std::vector<int> secondpsflayers = {};
    // std::vector<int> secondpsfcubes = {};
    RangeMap<std::string> cubePSFMap;
    RangeMap<std::string> layerPSFMap;

    
    bool time; // for compatibility with deconv implementation, these dont belong here
    bool saveSubimages;
    std::string gpu = "";
    
    // bool loadFromJSON(const json& jsonData) override;

private:
    void registerRangeMap(const std::string& jsonTag, RangeMap<std::string>& field, bool optional);

    virtual void registerAllParameters() override;
};



