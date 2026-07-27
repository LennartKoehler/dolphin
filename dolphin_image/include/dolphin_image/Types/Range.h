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
#include <string>
#include <sstream>
#include <stdexcept>
#include <cassert>
#include <cstdint>

#include "nlohmann/json.hpp"
#include <spdlog/spdlog.h>

using json = nlohmann::json;

template<typename T>
struct Range {
    size_t start;
    size_t end;  // SIZE_MAX means open-ended
    std::vector<T> values;

    // Default constructor
    Range() : start(0), end(0) {}

    Range(size_t s, size_t e, T value) : start(s), end(e) {
        values.push_back(std::forward<T>(value));
    }
    Range(size_t s, size_t e, std::vector<T> values) : start(s), end(e), values(values){}

    bool contains(size_t index) const {
        if (index == start && index == end){ return true; } // for the case of not a range, but just an index
        return index >= start && (end == SIZE_MAX || index < end);
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
    virtual void addRange(size_t start, size_t end, T value) {
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

    std::vector<Range<T>> get(size_t index) const {
        std::vector<Range<T>> result;

        for (const auto& range : ranges) {
            if (range.contains(index)) {
                result.push_back(range);
            }
        }

        return result;
    }

    // Get pointers to the values without copying
    std::vector<const T*> getPointers(size_t index) const {
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

    Range<T> operator[](size_t index) const {
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
            } catch (const std::exception&) {
                // spdlog::error("Failed to parse string as JSON or custom format: {}", e.what());
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
                std::pair<size_t, size_t> rangePair = parseRange(rangeKey);
                size_t start = rangePair.first;
                size_t end = rangePair.second;

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
                spdlog::error("Failed to parse range '{}': {}", rangeKey, e.what());
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
    std::vector<iterator> findRangesContaining(size_t index) {
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
        spdlog::info("[DEBUG] RangeMap contents:");
        for (const auto& range : ranges) {
            spdlog::info("[DEBUG] Range {}:", range.start);
            if (range.end == SIZE_MAX) {
                spdlog::info("END");
            } else {
                spdlog::info("{}", range.end);
            }
            spdlog::info(" -> ");
            for (const auto& value : range.values) {
                spdlog::info("'{}' ", value);
            }
            spdlog::info("");
        }
    }

private:
    // Fixed parseRange function
    std::pair<size_t, size_t> parseRange(const std::string& rangeStr) const {
        assert (!rangeStr.empty() && "Trying to parse empty string");
        size_t colonPos = rangeStr.find(':');
        size_t start, end;

        if (colonPos == std::string::npos) {
            start = static_cast<size_t>(std::stoull(rangeStr));
            end = start;
            return {start, end};
        }


        // Handle start
        if (colonPos == 0) {
            start = 0;
        } else {
            start = static_cast<size_t>(std::stoull(rangeStr.substr(0, colonPos)));
        }

        // Handle end
        if (colonPos == rangeStr.length() - 1) {
            end = SIZE_MAX; // Open-ended
        } else {
            std::string endStr = rangeStr.substr(colonPos + 1);
            if (endStr.empty()) {
                end = SIZE_MAX;
            } else {
                end = static_cast<size_t>(std::stoull(endStr));
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
