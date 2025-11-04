#pragma once
#include <vector>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include "psf/PSF.h"

struct BoxCoord {
    int x, y, z;
    int width, height, depth;
};
template<typename entryType>
struct BoxEntryPair {
    BoxCoord box;
    std::vector<entryType> entry;
    
    BoxEntryPair(BoxCoord b, std::vector<entryType> p) 
        : box(b), entry(std::move(p)) {}
};

template<typename entryType>
class ImageMap {

private:
    std::vector<BoxEntryPair<entryType>> entries_;

public:
    // Add mapping with any PSF type (ID string or actual PSF object)
    void add(const BoxCoord& box, const std::vector<entryType>& entry) {
        entries_.emplace_back(box, entry);
    }

    BoxEntryPair<entryType>& get(int index){
        assert(index < size() && "BoxEntryPair out of range");
        return entries_[index];
    }
    // Iterator support
    auto begin() { return entries_.begin(); }
    auto end() { return entries_.end(); }
    auto begin() const { return entries_.begin(); }
    auto end() const { return entries_.end(); }
    
    size_t size() const { return entries_.size(); }
    bool empty() const { return entries_.empty(); }
};

// static PSFObjectMap resolve(const PSFIDMap& idMap, 
//                             const std::vector<PSF>& availablePSFs) {
//     // Create lookup table
//     std::unordered_map<std::string, const PSF*> psfLookup;
//     for (const auto& psf : availablePSFs) {
//         psfLookup[psf.ID] = &psf;
//     }
    
//     PSFObjectMap objectMap;
    
//     for (const auto& entry : idMap) {
//         std::vector<PSF> resolvedPSFs;
//         for (const std::string& id : entry.psfs) {
//             auto it = psfLookup.find(id);
//             if (it != psfLookup.end()) {
//                 resolvedPSFs.push_back(*it->second);
//             } else {
//                 throw std::runtime_error("PSF with ID '" + id + "' not found");
//             }
//         }
//         objectMap.add(entry.box, resolvedPSFs);
//     }
    
//     return objectMap;
// }

